"""Code-aware chunking with AST-first and regex-fallback strategies."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

from rag_core.config.env_access import get_env as config_get_env

from .protocol import Chunk, ChunkConfig

logger = logging.getLogger(__name__)

# Patterns keyed by language family for regex fallback.
_LANGUAGE_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^\s*(class\s+|def\s+|async\s+def\s+)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"^\s*(export\s+)?(async\s+)?function\s+", re.MULTILINE),
        re.compile(r"^\s*(export\s+)?class\s+", re.MULTILINE),
        re.compile(
            r"^\s*(export\s+)?(const|let|var)\s+[\w$]+\s*=\s*(async\s*)?\([^\)]*\)\s*=>",
            re.MULTILINE,
        ),
    ],
    "go": [re.compile(r"^\s*func\s+", re.MULTILINE)],
    "rust": [re.compile(r"^\s*(pub\s+)?(fn|impl|struct|enum)\s+", re.MULTILINE)],
    "java": [
        re.compile(r"^\s*(public|private|protected)?\s*(static\s+)?class\s+", re.MULTILINE),
        re.compile(
            r"^\s*(public|private|protected)?\s*(static\s+)?[\w<>\[\]]+\s+\w+\s*\([^\)]*\)\s*\{",
            re.MULTILINE,
        ),
    ],
    "c": [
        re.compile(r"^\s*(typedef\s+)?(struct|enum|union)\s+\w+", re.MULTILINE),
        re.compile(
            r"^\s*(static\s+|inline\s+|extern\s+)?[\w\*\s]+\s+\w+\s*\([^\)]*\)\s*\{",
            re.MULTILINE,
        ),
    ],
    "cpp": [
        re.compile(r"^\s*(class|struct|namespace|template)\s+\w+", re.MULTILINE),
        re.compile(r"^\s*[\w:\<\>\~\*&\s]+\s+\w+\s*\([^\)]*\)\s*(const\s*)?\{", re.MULTILINE),
    ],
    "csharp": [
        re.compile(
            r"^\s*(public|private|protected|internal)?\s*(class|struct|interface|enum)\s+",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(public|private|protected|internal)\s+[\w<>\[\],\?]+\s+\w+\s*\([^\)]*\)\s*\{",
            re.MULTILINE,
        ),
    ],
    "ruby": [
        re.compile(r"^\s*(class|module|def)\s+", re.MULTILINE),
    ],
    "php": [
        re.compile(r"^\s*(class|trait|interface)\s+", re.MULTILINE),
        re.compile(r"^\s*(public|private|protected)?\s*function\s+\w+\s*\(", re.MULTILINE),
    ],
    "swift": [
        re.compile(r"^\s*(class|struct|enum|protocol|extension|func)\s+", re.MULTILINE),
    ],
    "kotlin": [
        re.compile(
            r"^\s*(class|data\s+class|sealed\s+class|object|interface|fun)\s+",
            re.MULTILINE,
        ),
    ],
    "scala": [
        re.compile(r"^\s*(class|object|trait|def)\s+", re.MULTILINE),
    ],
    "terraform": [
        re.compile(
            r'^\s*(resource|module|variable|output|provider|data|locals|terraform)\s+"',
            re.MULTILINE,
        ),
    ],
}

_FALLBACK_PATTERNS = [
    re.compile(r"^\s*(class\s+|def\s+|async\s+def\s+)", re.MULTILINE),
    re.compile(r"^\s*(export\s+)?(async\s+)?function\s+", re.MULTILINE),
    re.compile(r"^\s*func\s+", re.MULTILINE),
    re.compile(r"^\s*(pub\s+)?(fn|impl|struct|enum)\s+", re.MULTILINE),
    re.compile(r"^\s*(class|module|interface|trait|resource)\s+", re.MULTILINE),
]

_TREE_SITTER_LANGUAGE_CANDIDATES: Dict[str, Sequence[str]] = {
    "python": ("python",),
    "javascript": ("javascript", "typescript", "tsx"),
    "java": ("java",),
    "go": ("go",),
    "rust": ("rust",),
    "c": ("c",),
    "cpp": ("cpp", "c++"),
    "csharp": ("c_sharp", "csharp"),
    "ruby": ("ruby",),
    "php": ("php",),
    "swift": ("swift",),
    "kotlin": ("kotlin",),
    "scala": ("scala",),
    "terraform": ("hcl", "terraform"),
}

_MAGIKA_TO_INTERNAL_LANGUAGE: Dict[str, str] = {
    "c++": "cpp",
    "c#": "csharp",
    "js": "javascript",
    "ts": "javascript",
}


def _env_flag(name: str, *, default: bool) -> bool:
    raw = config_get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _mask_non_code_regions(text: str) -> str:
    """Mask comments and quoted strings so boundary regexes avoid false positives."""

    def _spaces(match: re.Match[str]) -> str:
        return " " * len(match.group(0))

    masked = text
    masked = re.sub(r'"""[\s\S]*?"""', _spaces, masked)
    masked = re.sub(r"'''[\s\S]*?'''", _spaces, masked)
    masked = re.sub(r"(?m)^\s*#.*$", _spaces, masked)
    masked = re.sub(r"(?m)^\s*//.*$", _spaces, masked)
    masked = re.sub(r"/\*[\s\S]*?\*/", _spaces, masked)
    masked = re.sub(r'"(?:\\.|[^"\\])*"', _spaces, masked)
    masked = re.sub(r"'(?:\\.|[^'\\])*'", _spaces, masked)
    return masked


class CodeChunker:
    """Chunks source code by preferring AST boundaries and falling back to regex."""

    def __init__(
        self,
        language: Optional[str] = None,
        *,
        skip_unsupported_language: Optional[bool] = None,
        enable_magika_detection: Optional[bool] = None,
    ) -> None:
        self._language = language.lower() if language else None
        self._skip_unsupported_language = (
            _env_flag("CHUNKING_SKIP_UNSUPPORTED_CODE", default=False)
            if skip_unsupported_language is None
            else skip_unsupported_language
        )
        self._enable_magika_detection = (
            _env_flag("CHUNKING_ENABLE_MAGIKA_LANGUAGE_DETECTION", default=True)
            if enable_magika_detection is None
            else enable_magika_detection
        )
        self._magika = None

    def _resolve_bounds(
        self,
        full_text: str,
        chunk_text: str,
        *,
        search_start: int,
    ) -> tuple[int, int]:
        if not chunk_text:
            return search_start, search_start

        probe = chunk_text[:80]
        start = full_text.find(probe, search_start)
        if start < 0:
            start = search_start

        end = min(len(full_text), start + len(chunk_text))
        return start, end

    def _language_candidates(self, language: str) -> Sequence[str]:
        return _TREE_SITTER_LANGUAGE_CANDIDATES.get(language, (language,))

    def _tree_sitter_backend_available(self) -> bool:
        try:
            from tree_sitter_language_pack import get_parser  # noqa: F401
        except ImportError:
            # Tree-sitter is optional; callers can still use regex chunking.
            return False
        return True

    def _get_tree_sitter_parser(self, language: str) -> Optional[object]:
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            # Tree-sitter is optional; callers can still use regex chunking.
            return None

        for candidate in self._language_candidates(language):
            try:
                return get_parser(candidate)
            except Exception:
                # Candidate names differ across installed language packs, so keep probing aliases.
                continue
        return None

    def _detect_language_with_magika(self, text: str) -> Optional[str]:
        if not self._enable_magika_detection:
            return None

        try:
            from magika import Magika
        except ImportError:
            # Language detection is advisory only; chunking can proceed without it.
            return None

        if self._magika is None:
            self._magika = Magika()

        try:
            result = self._magika.identify_bytes(text.encode("utf-8", errors="ignore"))
            label = str(result.output.label).lower()
        except Exception:
            # Detection failures should not block chunking when the content is otherwise valid.
            return None

        return _MAGIKA_TO_INTERNAL_LANGUAGE.get(label, label)

    def _resolve_language(self, text: str) -> Optional[str]:
        if self._language:
            return self._language
        return self._detect_language_with_magika(text)

    def _ast_boundaries_for_language(
        self, text: str, language: Optional[str]
    ) -> Optional[List[int]]:
        if not language:
            return None

        parser = self._get_tree_sitter_parser(language)
        if parser is None:
            return None

        try:
            tree = parser.parse(text.encode("utf-8", errors="ignore"))
            root = tree.root_node
        except Exception:
            # Parser errors should fall back to regex boundaries instead of dropping the document.
            return None

        boundaries = {0}
        named_children = getattr(root, "named_children", None)
        children = named_children if named_children else getattr(root, "children", [])

        for child in children:
            start = int(getattr(child, "start_byte", 0))
            end = int(getattr(child, "end_byte", 0))
            if end - start < 8:
                continue
            child_type = str(getattr(child, "type", ""))
            if child_type == "comment":
                continue
            boundaries.add(start)

        result = sorted(boundaries)
        return result if len(result) > 1 else None

    def _regex_boundaries(self, text: str, language: Optional[str]) -> List[int]:
        patterns = _LANGUAGE_PATTERNS.get(language or "", _FALLBACK_PATTERNS)
        masked = _mask_non_code_regions(text)
        boundaries = {0}

        for pattern in patterns:
            for match in pattern.finditer(masked):
                boundaries.add(match.start())

        return sorted(boundaries)

    def _segments_from_boundaries(self, text: str, boundaries: List[int]) -> List[str]:
        segments: List[str] = []
        for index, start in enumerate(boundaries):
            end = boundaries[index + 1] if index + 1 < len(boundaries) else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
        return segments

    def _flush_buffer(
        self,
        *,
        text: str,
        chunks: List[Chunk],
        buffer: List[str],
        index: int,
        search_start: int,
        metadata: Dict[str, str],
        joiner: str,
    ) -> tuple[int, int]:
        chunk_text = joiner.join(buffer).strip()
        if not chunk_text:
            return index, search_start

        start_char, end_char = self._resolve_bounds(
            text,
            chunk_text,
            search_start=search_start,
        )
        chunks.append(
            Chunk(
                text=chunk_text,
                index=index,
                start_char=start_char,
                end_char=end_char,
                metadata=dict(metadata),
            )
        )
        return index + 1, end_char

    def _retain_overlap(self, buffer: List[str], overlap: int) -> tuple[List[str], int]:
        if overlap <= 0 or not buffer:
            return [], 0

        last = buffer[-1]
        tail = last[-overlap:] if len(last) > overlap else last
        return [tail], len(tail)

    def _build_chunk_metadata(
        self,
        chunking_engine: str,
        resolved_language: Optional[str],
    ) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            "chunking_strategy": "code",
            "chunking_engine": chunking_engine,
        }
        if resolved_language:
            metadata["language"] = resolved_language
        return metadata

    def chunk(self, text: str, config: ChunkConfig) -> List[Chunk]:
        if not text:
            return []

        resolved_language = self._resolve_language(text)
        ast_boundaries = self._ast_boundaries_for_language(text, resolved_language)

        if (
            ast_boundaries is None
            and resolved_language
            and self._skip_unsupported_language
            and self._tree_sitter_backend_available()
        ):
            logger.info(
                "Skipping code chunking for unsupported tree-sitter language '%s'",
                resolved_language,
            )
            return []

        chunking_engine = "ast" if ast_boundaries else "regex"
        boundaries = ast_boundaries or self._regex_boundaries(text, resolved_language)
        segments = self._segments_from_boundaries(text, boundaries)
        metadata = self._build_chunk_metadata(chunking_engine, resolved_language)

        chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_len = 0
        chunk_idx = 0
        search_start = 0

        for segment in segments:
            if buffer_len + len(segment) > config.max_chars and buffer:
                chunk_idx, search_start = self._flush_buffer(
                    text=text,
                    chunks=chunks,
                    buffer=buffer,
                    index=chunk_idx,
                    search_start=search_start,
                    metadata=metadata,
                    joiner="\n\n",
                )
                buffer, buffer_len = self._retain_overlap(buffer, config.overlap)

            if len(segment) > config.max_chars:
                lines = segment.split("\n")
                for line in lines:
                    if buffer_len + len(line) + 1 > config.max_chars and buffer:
                        chunk_idx, search_start = self._flush_buffer(
                            text=text,
                            chunks=chunks,
                            buffer=buffer,
                            index=chunk_idx,
                            search_start=search_start,
                            metadata=metadata,
                            joiner="\n",
                        )
                        buffer = []
                        buffer_len = 0

                    buffer.append(line)
                    buffer_len += len(line) + 1
                continue

            buffer.append(segment)
            buffer_len += len(segment) + 2

        if buffer:
            self._flush_buffer(
                text=text,
                chunks=chunks,
                buffer=buffer,
                index=chunk_idx,
                search_start=search_start,
                metadata=metadata,
                joiner="\n\n",
            )

        return chunks
