from __future__ import annotations

from rag_core.documents.chunking.router import chunk_text
from rag_core.search.types import ChunkResult


def _is_code(mime_type: str | None, filename: str | None) -> bool:
    lowered_mime = (mime_type or '').strip().lower()
    lowered_filename = (filename or '').strip().lower()
    if lowered_mime.startswith('text/x-'):
        return True
    return lowered_filename.endswith(
        (
            '.py',
            '.js',
            '.jsx',
            '.ts',
            '.tsx',
            '.java',
            '.go',
            '.rs',
            '.rb',
            '.php',
            '.swift',
            '.kt',
            '.scala',
            '.c',
            '.cc',
            '.cpp',
            '.h',
            '.hpp',
            '.cs',
            '.sql',
            '.tf',
        )
    )


def chunk_content(
    text: str,
    mime_type: str | None = None,
    filename: str | None = None,
    content_bytes: bytes | None = None,
) -> list[ChunkResult]:
    chunks = chunk_text(
        text,
        mime_type=mime_type,
        filename=filename,
    )
    return [
        ChunkResult(
            text=chunk.text,
            start_index=chunk.start_char,
            end_index=chunk.end_char,
            token_count=max(0, len(chunk.text.split())),
        )
        for chunk in chunks
    ]
