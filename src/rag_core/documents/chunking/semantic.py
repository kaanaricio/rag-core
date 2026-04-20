"""Semantic chunking using embedding similarity to find topic boundaries."""

from __future__ import annotations

import asyncio
import logging
import math
from rag_core.config.env_access import get_env as config_get_env
import re
from typing import Awaitable, Callable, Dict, List, Optional

from .protocol import Chunk, ChunkConfig

logger = logging.getLogger(__name__)

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


def _env_flag(name: str, *, default: bool) -> bool:
    raw = config_get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = _SENTENCE_PATTERN.split(text)
    result: List[str] = []
    for sentence in sentences:
        parts = sentence.split("\n\n")
        result.extend(part.strip() for part in parts if part.strip())
    return result


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _LocalSemanticEmbedder:
    """Singleton local sentence embedder."""

    _instances: Dict[str, "_LocalSemanticEmbedder"] = {}

    @classmethod
    def get(cls, model_name: str) -> "_LocalSemanticEmbedder":
        instance = cls._instances.get(model_name)
        if instance is None:
            instance = cls(model_name)
            cls._instances[model_name] = instance
        return instance

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed_many(self, sentences: List[str]) -> List[List[float]]:
        if not sentences:
            return []

        def _encode() -> List[List[float]]:
            model = self._load_model()
            vectors = model.encode(sentences, show_progress_bar=False)
            return [list(map(float, vector)) for vector in vectors]

        return await asyncio.to_thread(_encode)


class SemanticChunker:
    """Chunks text by finding semantic boundaries using embedding similarity."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.75,
        embed_fn: Optional[EmbedFn] = None,
        enable_local_model: Optional[bool] = None,
        local_model_name: Optional[str] = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._embed_fn = embed_fn
        self._enable_local_model = (
            _env_flag("CHUNKING_ENABLE_LOCAL_SEMANTIC", default=False)
            if enable_local_model is None
            else enable_local_model
        )
        configured_model = config_get_env("CHUNKING_SEMANTIC_LOCAL_MODEL", "").strip()
        self._local_model_name = (
            local_model_name or configured_model or "sentence-transformers/all-MiniLM-L6-v2"
        )

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

    def _segment_to_chunks(self, segment: str, *, config: ChunkConfig) -> List[str]:
        if len(segment) <= config.max_chars:
            return [segment]

        pieces: List[str] = []
        step = max(1, config.max_chars - max(0, config.overlap))
        index = 0
        while index < len(segment):
            piece = segment[index : index + config.max_chars].strip()
            if piece:
                pieces.append(piece)
            index += step
        return pieces

    def _get_local_embed_fn(self) -> Optional[EmbedFn]:
        if not self._enable_local_model:
            return None

        try:
            embedder = _LocalSemanticEmbedder.get(self._local_model_name)
        except Exception as exc:
            logger.warning("Local semantic embedder setup failed: %s", exc)
            return None

        async def _embed(sentences: List[str]) -> List[List[float]]:
            return await embedder.embed_many(sentences)

        return _embed

    def _resolve_embed_fn(self) -> Optional[EmbedFn]:
        if self._embed_fn is not None:
            return self._embed_fn
        return self._get_local_embed_fn()

    def _build_chunks_from_segments(
        self,
        text: str,
        segments: List[str],
        config: ChunkConfig,
        *,
        strategy_name: str,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_idx = 0
        search_start = 0

        for segment in segments:
            for piece in self._segment_to_chunks(segment, config=config):
                start_char, end_char = self._resolve_bounds(
                    text,
                    piece,
                    search_start=search_start,
                )
                search_start = end_char
                chunks.append(
                    Chunk(
                        text=piece,
                        index=chunk_idx,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"chunking_strategy": strategy_name},
                    )
                )
                chunk_idx += 1

        return chunks

    def chunk(self, text: str, config: ChunkConfig) -> List[Chunk]:
        """Synchronous path using heuristic boundaries only."""
        if not text:
            return []

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text, index=0, start_char=0, end_char=len(text))]

        return self._paragraph_heuristic_chunk(text, sentences, config)

    async def chunk_async(self, text: str, config: ChunkConfig) -> List[Chunk]:
        """Async semantic chunking with embedding-based boundaries."""
        if not text:
            return []

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text, index=0, start_char=0, end_char=len(text))]

        embed_fn = self._resolve_embed_fn()
        if embed_fn is None:
            return self._paragraph_heuristic_chunk(text, sentences, config)

        try:
            embeddings = await embed_fn(sentences)
        except Exception as exc:
            logger.warning("Semantic embedding failed, using heuristic fallback: %s", exc)
            return self._paragraph_heuristic_chunk(text, sentences, config)

        if len(embeddings) != len(sentences):
            logger.warning(
                "Semantic embedding length mismatch (%d != %d), using heuristic fallback",
                len(embeddings),
                len(sentences),
            )
            return self._paragraph_heuristic_chunk(text, sentences, config)

        boundaries: List[int] = [0]
        for idx in range(1, len(embeddings)):
            similarity = _cosine_similarity(embeddings[idx - 1], embeddings[idx])
            if similarity < self._threshold:
                boundaries.append(idx)

        segments: List[str] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            chunk_text = " ".join(sentences[start:end]).strip()
            if chunk_text:
                segments.append(chunk_text)

        return self._build_chunks_from_segments(
            text,
            segments,
            config,
            strategy_name="semantic",
        )

    def _paragraph_heuristic_chunk(
        self,
        full_text: str,
        sentences: List[str],
        config: ChunkConfig,
    ) -> List[Chunk]:
        """Heuristic fallback: group sentences by max_chars and overlap only."""
        chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_len = 0
        chunk_idx = 0
        search_start = 0

        for sentence in sentences:
            if buffer_len + len(sentence) + 1 > config.max_chars and buffer:
                chunk_text = " ".join(buffer).strip()
                start_char, end_char = self._resolve_bounds(
                    full_text,
                    chunk_text,
                    search_start=search_start,
                )
                search_start = end_char
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=chunk_idx,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"chunking_strategy": "semantic_heuristic"},
                    )
                )
                chunk_idx += 1

                if config.overlap > 0:
                    overlap_text = chunk_text[-config.overlap :]
                    buffer = [overlap_text]
                    buffer_len = len(overlap_text)
                else:
                    buffer = []
                    buffer_len = 0

            buffer.append(sentence)
            buffer_len += len(sentence) + 1

        if buffer:
            chunk_text = " ".join(buffer).strip()
            if chunk_text:
                start_char, end_char = self._resolve_bounds(
                    full_text,
                    chunk_text,
                    search_start=search_start,
                )
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=chunk_idx,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"chunking_strategy": "semantic_heuristic"},
                    )
                )

        return chunks
