from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_core.search.chunking import _is_code, chunk_content
from rag_core.search.text_builder import build_sparse_text, build_textual_representation
from rag_core.search.types import (
    ChunkResult,
    ContentType,
    EmbeddingProvider,
    SparseEmbedder,
    SparseVector,
)

from .indexer_models import IndexRequest

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 50


@dataclass(frozen=True)
class PreparedIndexData:
    content_type: ContentType
    chunks: list[ChunkResult]
    dense_vectors: list[list[float]]
    payload_texts: list[str]
    sparse_channels: list[dict[str, SparseVector]]


async def prepare_index_data(
    *,
    req: IndexRequest,
    embedding_provider: EmbeddingProvider,
    sparse_embedder: SparseEmbedder,
) -> PreparedIndexData:
    content_type = _resolve_content_type(req.mime_type, req.filename)
    chunks = _resolve_chunks(req)
    if not chunks:
        return PreparedIndexData(
            content_type=content_type,
            chunks=[],
            dense_vectors=[],
            payload_texts=[],
            sparse_channels=[],
        )

    embedding_source_chunks = _resolve_embedding_source_chunks(req, chunks)
    dense_texts = _build_dense_texts(
        req=req,
        content_type=content_type,
        embedding_source_chunks=embedding_source_chunks,
    )
    payload_texts = _build_payload_texts(
        req=req,
        content_type=content_type,
        chunks=chunks,
    )
    sparse_texts = _build_sparse_texts(req=req, chunks=chunks)

    dense_vectors = await _embed_dense_texts(embedding_provider, dense_texts)
    sparse_channels = _embed_sparse_channels(
        sparse_embedder=sparse_embedder,
        texts=sparse_texts,
        expected_count=len(chunks),
    )
    return PreparedIndexData(
        content_type=content_type,
        chunks=chunks,
        dense_vectors=dense_vectors,
        payload_texts=payload_texts,
        sparse_channels=sparse_channels,
    )


def _resolve_content_type(mime_type: str, filename: str) -> ContentType:
    return ContentType.CODE if _is_code(mime_type, filename) else ContentType.DOCUMENT


def _resolve_chunks(req: IndexRequest) -> list[ChunkResult]:
    if req.pre_chunked_texts:
        return [
            ChunkResult(text=text, start_index=0, end_index=len(text), token_count=0)
            for text in req.pre_chunked_texts
        ]
    return chunk_content(
        text=req.text,
        mime_type=req.mime_type,
        filename=req.filename,
        content_bytes=req.content_bytes,
    )


def _resolve_embedding_source_chunks(
    req: IndexRequest,
    chunks: list[ChunkResult],
) -> list[str]:
    if req.embedding_chunk_texts and len(req.embedding_chunk_texts) == len(chunks):
        return list(req.embedding_chunk_texts)
    return [chunk.text for chunk in chunks]


def _build_dense_texts(
    *,
    req: IndexRequest,
    content_type: ContentType,
    embedding_source_chunks: list[str],
) -> list[str]:
    return [
        build_textual_representation(
            content=text,
            source_type=req.source_type,
            name=req.filename,
            content_type=content_type,
            path=req.path,
            extra_fields=req.extra_fields,
        )
        for text in embedding_source_chunks
    ]


def _build_payload_texts(
    *,
    req: IndexRequest,
    content_type: ContentType,
    chunks: list[ChunkResult],
) -> list[str]:
    return [
        build_textual_representation(
            content=chunk.text,
            source_type=req.source_type,
            name=req.filename,
            content_type=content_type,
            path=req.path,
            extra_fields=req.extra_fields,
        )
        for chunk in chunks
    ]


def _build_sparse_texts(
    *,
    req: IndexRequest,
    chunks: list[ChunkResult],
) -> list[str]:
    metadata: dict[str, str] = {
        "source_type": req.source_type,
        "filename": req.filename,
        "document_id": req.document_id,
    }
    if req.extra_fields:
        metadata.update(req.extra_fields)
    return [
        build_sparse_text(chunk_text=chunk.text, metadata=metadata)
        for chunk in chunks
    ]


async def _embed_dense_texts(
    embedding_provider: EmbeddingProvider,
    texts: list[str],
) -> list[list[float]]:
    dense_vectors: list[list[float]] = []
    for index in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[index : index + _EMBED_BATCH_SIZE]
        dense_vectors.extend(await embedding_provider.embed_texts(batch))
    return dense_vectors


def _embed_sparse_channels(
    *,
    sparse_embedder: SparseEmbedder,
    texts: list[str],
    expected_count: int,
) -> list[dict[str, SparseVector]]:
    sparse_channels = _try_embed_sparse_multi(sparse_embedder=sparse_embedder, texts=texts)
    if sparse_channels is None:
        sparse_vectors = sparse_embedder.embed_texts(texts)
        sparse_channels = [{"bm25": vector} for vector in sparse_vectors]
    if len(sparse_channels) != expected_count:
        raise ValueError(
            "Sparse embedding count mismatch: expected %d got %d"
            % (expected_count, len(sparse_channels))
        )
    return sparse_channels


def _try_embed_sparse_multi(
    *,
    sparse_embedder: SparseEmbedder,
    texts: list[str],
) -> list[dict[str, SparseVector]] | None:
    embed_multi = getattr(sparse_embedder, "embed_texts_multi", None)
    if not callable(embed_multi):
        return None
    try:
        raw = embed_multi(texts)
    except Exception:
        logger.warning("Multi-channel sparse embedding failed; using bm25 only", exc_info=True)
        return None
    if not isinstance(raw, list) or len(raw) != len(texts):
        return None

    sparse_channels: list[dict[str, SparseVector]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        channel_map: dict[str, SparseVector] = {}
        for name, vector in item.items():
            if isinstance(name, str) and name and isinstance(vector, SparseVector):
                channel_map[name] = vector
        if not channel_map:
            return None
        sparse_channels.append(channel_map)
    return sparse_channels
