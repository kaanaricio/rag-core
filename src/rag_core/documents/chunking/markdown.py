"""Markdown-aware chunking strategy."""

from __future__ import annotations

from typing import List

from .protocol import Chunk, ChunkConfig


class MarkdownChunker:
    """Chunks markdown by splitting on headers, then paragraphs."""

    def chunk(self, text: str, config: ChunkConfig) -> List[Chunk]:
        if not text:
            return []

        max_chars = config.max_chars
        overlap = config.overlap

        lines = text.splitlines()
        sections: List[str] = []
        current: List[str] = []

        for line in lines:
            if line.strip().startswith("#") and current:
                sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current).strip())

        raw_chunks: List[str] = []
        for section in sections:
            if len(section) <= max_chars:
                raw_chunks.append(section)
                continue

            parts = [p.strip() for p in section.split("\n\n") if p.strip()]
            buffer: List[str] = []
            buffer_len = 0
            for part in parts:
                if buffer_len + len(part) + 2 > max_chars and buffer:
                    chunk_text = "\n\n".join(buffer).strip()
                    raw_chunks.append(chunk_text)
                    if overlap > 0 and chunk_text:
                        overlap_text = chunk_text[-overlap:]
                        buffer = [overlap_text]
                        buffer_len = len(overlap_text)
                    else:
                        buffer = []
                        buffer_len = 0
                buffer.append(part)
                buffer_len += len(part) + 2
            if buffer:
                raw_chunks.append("\n\n".join(buffer).strip())

        # Convert to Chunk objects with positions
        chunks: List[Chunk] = []
        char_pos = 0
        for idx, raw in enumerate(raw_chunks):
            if not raw:
                continue
            start = text.find(raw[:50], char_pos)
            if start == -1:
                start = char_pos
            chunks.append(
                Chunk(
                    text=raw,
                    index=idx,
                    start_char=start,
                    end_char=start + len(raw),
                )
            )
            char_pos = start + len(raw)

        return chunks
