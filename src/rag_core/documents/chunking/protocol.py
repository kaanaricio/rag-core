"""Chunking strategy protocol and types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Union


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for chunking."""

    max_chars: int = 2000
    overlap: int = 200
    strategy: str = "auto"  # "auto", "markdown", "semantic", "code"


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Union[str, int, float, bool, None]]] = field(
        default=None
    )


class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self, text: str, config: ChunkConfig) -> List[Chunk]: ...


class AsyncChunkingStrategy(Protocol):
    """Protocol for async chunking strategies."""

    async def chunk_async(self, text: str, config: ChunkConfig) -> List[Chunk]: ...
