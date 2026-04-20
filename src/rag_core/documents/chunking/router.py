"""Chunking strategy router - selects strategy based on content type."""

from __future__ import annotations

from rag_core.config.env_access import get_env as config_get_env
from typing import Awaitable, Callable, List, Optional

from .code import CodeChunker
from .markdown import MarkdownChunker
from .protocol import Chunk, ChunkConfig
from .semantic import SemanticChunker

DEFAULT_STRATEGY = config_get_env("CHUNKING_STRATEGY", "auto")

_CODE_MIMES = {
    "text/x-python",
    "text/javascript",
    "text/typescript",
    "application/javascript",
    "text/x-csrc",
    "text/x-c++src",
    "text/x-csharp",
    "text/x-java",
    "text/x-c",
    "text/x-go",
    "text/x-rust",
    "text/x-kotlin",
    "text/x-scala",
    "text/x-ruby",
    "application/x-httpd-php",
    "text/x-swift",
    "application/x-terraform",
}

_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".hpp",
    ".h",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".tf",
    ".tfvars",
}

_MIME_TO_LANGUAGE = {
    "text/x-python": "python",
    "text/javascript": "javascript",
    "application/javascript": "javascript",
    "text/typescript": "javascript",
    "text/x-csrc": "c",
    "text/x-c++src": "cpp",
    "text/x-csharp": "csharp",
    "text/x-java": "java",
    "text/x-go": "go",
    "text/x-rust": "rust",
    "text/x-kotlin": "kotlin",
    "text/x-scala": "scala",
    "text/x-ruby": "ruby",
    "application/x-httpd-php": "php",
    "text/x-swift": "swift",
    "application/x-terraform": "terraform",
}

_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".tf": "terraform",
    ".tfvars": "terraform",
}


EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


def chunk_text(
    text: str,
    *,
    config: Optional[ChunkConfig] = None,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[Chunk]:
    """Route to the appropriate chunking strategy."""
    if config is None:
        config = ChunkConfig()

    strategy = config.strategy
    if strategy == "auto":
        strategy = _detect_strategy(text, mime_type=mime_type, filename=filename)

    if strategy == "code":
        language = _detect_code_language(mime_type=mime_type, filename=filename)
        return CodeChunker(language=language).chunk(text, config)
    if strategy == "semantic":
        return SemanticChunker().chunk(text, config)
    return MarkdownChunker().chunk(text, config)


async def chunk_text_async(
    text: str,
    *,
    config: Optional[ChunkConfig] = None,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
    embed_fn: Optional[EmbedFn] = None,
) -> List[Chunk]:
    """Async variant that enables real semantic chunking with embedding calls."""
    if config is None:
        config = ChunkConfig()

    strategy = config.strategy
    if strategy == "auto":
        strategy = _detect_strategy(text, mime_type=mime_type, filename=filename)

    if strategy == "semantic":
        return await SemanticChunker(embed_fn=embed_fn).chunk_async(text, config)

    # Non-semantic strategies are synchronous.
    return chunk_text(
        text,
        config=config,
        mime_type=mime_type,
        filename=filename,
    )


def _detect_strategy(
    text: str,
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Auto-detect the best chunking strategy."""
    if mime_type and mime_type in _CODE_MIMES:
        return "code"

    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in _CODE_EXTENSIONS:
            return "code"

    lines = text[:5000].split("\n")
    code_indicators = sum(
        1
        for line in lines
        if line.strip().startswith(
            (
                "def ",
                "class ",
                "function ",
                "import ",
                "from ",
                "const ",
                "let ",
                "var ",
                "fn ",
                "interface ",
                "module ",
                "resource ",
            )
        )
    )
    if code_indicators > len(lines) * 0.1:
        return "code"

    return "markdown"


def _detect_code_language(
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
    if mime_type and mime_type in _MIME_TO_LANGUAGE:
        return _MIME_TO_LANGUAGE[mime_type]
    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        return _EXT_TO_LANGUAGE.get(ext)
    return None
