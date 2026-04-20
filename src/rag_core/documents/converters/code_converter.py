"""Code file converter with language detection metadata.

Improvement over AirWeave: adds language detection metadata
for downstream AST-aware chunking.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict

from .base import BaseConverter, ConversionResult, score_text_quality

logger = logging.getLogger(__name__)

# Extension to language mapping (comprehensive)
LANGUAGE_MAP: Dict[str, str] = {
    # Web/Frontend
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".svelte": "svelte",
    # Backend
    ".py": "python",
    ".rb": "ruby",
    ".php": "php",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".m": "objectivec",
    ".d": "d",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".clj": "clojure",
    ".groovy": "groovy",
    ".dart": "dart",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".fs": "fsharp",
    ".nim": "nim",
    ".cr": "crystal",
    ".zig": "zig",
    ".lua": "lua",
    ".pl": "perl",
    ".r": "r",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    # Config
    ".tf": "terraform",
    ".hcl": "hcl",
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    # Build
    ".gradle": "gradle",
    ".cmake": "cmake",
    ".make": "makefile",
    ".mak": "makefile",
}


def detect_language(filename: str) -> str:
    """Detect programming language from filename extension.

    Returns the language identifier for code fences, or 'text' if unknown.
    """
    _, ext = os.path.splitext(filename.lower())
    return LANGUAGE_MAP.get(ext, "text")


class CodeConverter(BaseConverter):
    """Converts code files to markdown with language detection metadata.

    Unlike AirWeave which returns raw UTF-8, we add language detection
    metadata for downstream AST-aware chunking.
    """

    format_name = "code"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert code file with language detection."""

        def _convert() -> ConversionResult:
            if not file_bytes:
                return ConversionResult(
                    metadata={"parser": "local:code"},
                    quality=score_text_quality(""),
                )

            # Try UTF-8 decode (most code files)
            try:
                code = file_bytes.decode("utf-8")
                if "\ufffd" in code:
                    raise UnicodeDecodeError("utf-8", b"", 0, 1, "replacement chars")
            except UnicodeDecodeError:
                # Fallback with replacement
                code = file_bytes.decode("utf-8", errors="replace")
                replacement_count = code.count("\ufffd")
                if replacement_count > 0:
                    logger.warning(
                        "Code file %s has %d replacement chars, may be binary",
                        filename,
                        replacement_count,
                    )
                    return ConversionResult(
                        metadata={
                            "parser": "local:code",
                            "error": "binary content detected",
                        },
                    )

            language = detect_language(filename)
            quality = score_text_quality(code)

            metadata: Dict[str, str | bool] = {
                "parser": "local:code",
                "language": language,
                "needs_ocr": False,
            }

            return ConversionResult(
                content=code,
                metadata=metadata,  # type: ignore[arg-type]
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
