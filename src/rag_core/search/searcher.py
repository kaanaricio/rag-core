"""Unified search orchestrator with hybrid retrieval and optional reranking."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Awaitable, Optional

from rag_core.search.types import (
    EmbeddingProvider,
    RerankerProvider,
    SearchSidecar,
    SearchSidecarQuery,
    SearchQuery,
    SearchResult,
    SparseEmbedder,
    SparseVector,
    VectorStore,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """Parameters for a unified search."""

    query: str
    corpus_ids: list[str]
    namespace: str
    limit: int = 20
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    rerank: bool = False
    query_vector: list[float] | None = None
    query_sparse_vectors: dict[str, SparseVector] | None = None
    use_sidecar: bool = True


class SearchOrchestrator:
    """Orchestrates hybrid search and optional reranking."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedder: SparseEmbedder,
        vector_store: VectorStore,
        reranker: Optional[RerankerProvider] = None,
        sidecar: SearchSidecar | None = None,
    ) -> None:
        self._embedding = embedding_provider
        self._sparse = sparse_embedder
        self._store = vector_store
        self._reranker = reranker
        self._sidecar = sidecar

    async def search(self, req: SearchRequest) -> list[SearchResult]:
        """Execute unified search across all sources."""
        dense_vec = (
            req.query_vector
            if req.query_vector is not None
            else await self._embedding.embed_query(req.query)
        )
        sparse_vectors = (
            req.query_sparse_vectors
            if req.query_sparse_vectors is not None
            else await asyncio.to_thread(_embed_sparse_query, self._sparse, req.query)
        )
        primary_sparse = sparse_vectors.get("bm25")
        if primary_sparse is None and sparse_vectors:
            primary_sparse = next(iter(sparse_vectors.values()))
        if primary_sparse is None:
            raise RuntimeError("No sparse query vector generated")

        vector_query = SearchQuery(
            dense_vector=dense_vec,
            sparse_vector=primary_sparse,
            sparse_vectors=sparse_vectors,
            namespace=req.namespace,
            corpus_ids=req.corpus_ids,
            content_types=req.content_types,
            document_ids=req.document_ids,
            limit=req.limit,
        )

        results = await self._search_all(req=req, vector_query=vector_query)

        if req.rerank and self._reranker and results:
            try:
                reranked = await self._reranker.rerank(
                    req.query,
                    [result.text for result in results],
                    top_k=req.limit,
                )
            except Exception:
                logger.warning(
                    "Reranking failed; returning search results without reranking",
                    exc_info=True,
                )
            else:
                reranked_results = [
                    results[rr.index]
                    for rr in reranked
                    if isinstance(getattr(rr, "index", None), int) and 0 <= rr.index < len(results)
                ]
                if reranked_results:
                    results = reranked_results

        return results[: req.limit]

    async def check_health(self) -> dict[str, object]:
        """Check health of the underlying vector store."""
        return await self._store.check_health()

    async def _search_all(
        self,
        *,
        req: SearchRequest,
        vector_query: SearchQuery,
    ) -> list[SearchResult]:
        vector_task = self._store.search(vector_query)
        sidecar_task = self._build_sidecar_task(req)
        if sidecar_task is None:
            return await vector_task

        vector_results, sidecar_results = await asyncio.gather(
            vector_task,
            sidecar_task,
            return_exceptions=True,
        )
        if isinstance(vector_results, Exception):
            raise vector_results
        if isinstance(vector_results, BaseException):
            raise RuntimeError("Vector search failed with a non-exception BaseException")
        if isinstance(sidecar_results, Exception):
            logger.warning(
                "Search sidecar failed; returning vector-store results only: %s",
                sidecar_results,
            )
            return vector_results
        if isinstance(sidecar_results, BaseException):
            logger.warning(
                "Search sidecar failed with a non-exception BaseException; returning vector-store results only",
            )
            return vector_results

        return _merge_results(vector_results, sidecar_results)

    def _build_sidecar_task(
        self,
        req: SearchRequest,
    ) -> Awaitable[list[SearchResult]] | None:
        if self._sidecar is None or not req.use_sidecar:
            return None

        sidecar_query = SearchSidecarQuery(
            query=req.query,
            namespace=req.namespace,
            corpus_ids=req.corpus_ids,
            limit=req.limit,
            content_types=req.content_types,
            document_ids=req.document_ids,
        )
        return asyncio.create_task(self._sidecar.search(sidecar_query))


def _embed_sparse_query(
    sparse_embedder: SparseEmbedder,
    query: str,
) -> dict[str, SparseVector]:
    """Embed query into all available sparse channels (bm25 + splade)."""
    return sparse_embedder.embed_query_multi(query)


def _merge_results(
    vector_results: list[SearchResult],
    sidecar_results: list[SearchResult],
) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}
    ordered: list[SearchResult] = []

    for result in sidecar_results + vector_results:
        existing = merged.get(result.id)
        if existing is not None:
            merged[result.id] = _merge_duplicate_results(existing, result)
            continue
        merged[result.id] = result
        ordered.append(result)

    return [merged[result.id] for result in ordered]


def _merge_duplicate_results(
    preferred: SearchResult,
    fallback: SearchResult,
) -> SearchResult:
    return replace(
        preferred,
        score=max(preferred.score, fallback.score),
        document_id=preferred.document_id or fallback.document_id,
        corpus_id=preferred.corpus_id or fallback.corpus_id,
        document_key=preferred.document_key or fallback.document_key,
        content_sha256=preferred.content_sha256 or fallback.content_sha256,
        title=preferred.title or fallback.title,
        section_id=preferred.section_id or fallback.section_id,
        section_title=preferred.section_title or fallback.section_title,
        section_path=preferred.section_path or fallback.section_path,
        document_path=preferred.document_path or fallback.document_path,
        chunk_index=preferred.chunk_index if preferred.chunk_index is not None else fallback.chunk_index,
        chunk_word_count=(
            preferred.chunk_word_count
            if preferred.chunk_word_count is not None
            else fallback.chunk_word_count
        ),
        chunk_token_estimate=(
            preferred.chunk_token_estimate
            if preferred.chunk_token_estimate is not None
            else fallback.chunk_token_estimate
        ),
        embedding_model=preferred.embedding_model or fallback.embedding_model,
        chunker_strategy=preferred.chunker_strategy or fallback.chunker_strategy,
        result_type=preferred.result_type or fallback.result_type,
        figure_id=preferred.figure_id or fallback.figure_id,
        figure_thumbnail_url=preferred.figure_thumbnail_url or fallback.figure_thumbnail_url,
        metadata={**fallback.metadata, **preferred.metadata},
    )
