from __future__ import annotations

import logging

from .indexer_embeddings import prepare_index_data
from .indexer_models import IndexRequest, IndexResult
from .indexer_points import build_points, make_point_id
from rag_core.search.types import (
    DeleteFilter,
    EmbeddingProvider,
    SparseEmbedder,
    VectorStore,
)

logger = logging.getLogger(__name__)


class QdrantIndexer:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedder: SparseEmbedder,
        vector_store: VectorStore,
    ) -> None:
        self._embedding = embedding_provider
        self._sparse = sparse_embedder
        self._store = vector_store

    async def index_document(self, req: IndexRequest) -> IndexResult:
        namespace = req.namespace.strip()
        if not namespace:
            raise ValueError("namespace is required for indexing")

        prepared = await prepare_index_data(
            req=req,
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
        )
        if not prepared.chunks:
            if req.document_id:
                await self._store.delete(
                    DeleteFilter(
                        namespace=namespace,
                        corpus_id=req.corpus_id,
                        document_id=req.document_id,
                    )
                )
            return IndexResult(
                document_id=req.document_id,
                chunk_count=0,
                point_ids=[],
                point_payloads=[],
                document_key=req.document_key,
                content_sha256=req.content_sha256,
            )

        points, point_ids = build_points(
            req=req,
            namespace=namespace,
            prepared=prepared,
        )
        await self._store.upsert(points)
        stale_point_ids = _build_stale_point_ids(req, new_chunk_count=len(points))
        if stale_point_ids:
            await self._store.delete_point_ids(stale_point_ids)

        logger.info("Indexed %d chunks for document %s", len(points), req.document_id)
        return IndexResult(
            document_id=req.document_id,
            chunk_count=len(points),
            point_ids=point_ids,
            point_payloads=[dict(point.payload) for point in points],
            document_key=req.document_key,
            content_sha256=req.content_sha256,
        )

    async def delete_document(
        self,
        document_id: str,
        namespace: str,
        *,
        corpus_id: str,
    ) -> None:
        """Delete all chunks for a document from Qdrant."""
        namespace_scoped = namespace.strip()
        if not namespace_scoped:
            raise ValueError("namespace is required for delete_document")
        corpus_scoped = corpus_id.strip()
        if not corpus_scoped:
            raise ValueError("corpus_id is required for delete_document")
        await self._store.delete(
            DeleteFilter(
                namespace=namespace_scoped,
                corpus_id=corpus_scoped,
                document_id=document_id,
            ),
        )


def _build_stale_point_ids(req: IndexRequest, *, new_chunk_count: int) -> list[str]:
    if not req.document_id or req.existing_chunk_count is None:
        return []
    if req.existing_chunk_count <= new_chunk_count:
        return []
    return [
        make_point_id(
            namespace=req.namespace,
            corpus_id=req.corpus_id,
            document_id=req.document_id,
            chunk_index=index,
        )
        for index in range(new_chunk_count, req.existing_chunk_count)
    ]
