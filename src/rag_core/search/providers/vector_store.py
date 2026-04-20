"""Qdrant vector store with hardened writes, adaptive batching, and health checks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Sequence, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.types import (
    DeleteFilter,
    SearchQuery,
    SearchResult,
    StoredDocumentRecord,
    VectorPoint,
)

from .vector_store_search import build_point, build_prefetches, point_to_result
from .vector_store_runtime import extract_dense_vector_size, extract_sparse_vector_names
from .vector_store_shared import (
    _DENSE_VECTOR_NAME,
    _KNOWN_SPARSE_VECTOR_NAMES,
    WriteLatencyTracker,
    compute_write_params,
)
from .vector_store_write import (
    split_into_batches,
    upsert_with_fallback,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant vector store with remote or embedded/local Qdrant support."""

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        collection_name: str,
        location: str | None = None,
        dense_dimensions: int = 3072,
        quantization_enabled: bool = True,
    ) -> None:
        if bool(url) == bool(location):
            raise ValueError("Provide exactly one of url or location for QdrantVectorStore")

        self._is_local = location is not None
        if location:
            self._client = AsyncQdrantClient(location=location, timeout=120)
        elif api_key:
            self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=120)
        else:
            self._client = AsyncQdrantClient(url=url, timeout=120)
        self._collection = collection_name
        self._dimensions = dense_dimensions
        self._quantization_enabled = quantization_enabled
        self._available_sparse_vector_names = _KNOWN_SPARSE_VECTOR_NAMES

        max_concurrent, self._max_batch_size = compute_write_params(dense_dimensions)
        self._write_sem = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

        self._collection_ready = False
        self._collection_ready_lock = asyncio.Lock()

        self._latency = WriteLatencyTracker()
        logger.info(
            "QdrantVectorStore initialized: collection=%s, dims=%d, concurrency=%d, "
            "batch_size=%d, quantization=%s",
            collection_name,
            dense_dimensions,
            max_concurrent,
            self._max_batch_size,
            quantization_enabled,
        )

    async def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        await self._client.close()

    async def __aenter__(self) -> "QdrantVectorStore":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if self._collection_ready:
            return

        async with self._collection_ready_lock:
            if self._is_collection_ready():
                return
            collections = await self._client.get_collections()
            existing = {c.name for c in collections.collections}
            if self._collection in existing:
                await self._assert_collection_compatible()
                logger.info(
                    "Collection %s already exists, skipping creation",
                    self._collection,
                )
                self._collection_ready = True
                return

            await self._create_collection()
            self._collection_ready = True

    def _is_collection_ready(self) -> bool:
        return self._collection_ready

    async def _create_collection(self) -> None:
        """Create a new Qdrant collection with optimized settings."""
        hnsw_config = rest.HnswConfigDiff(ef_construct=100)

        quantization_config: Optional[rest.ScalarQuantization] = None
        if self._quantization_enabled:
            quantization_config = rest.ScalarQuantization(
                scalar=rest.ScalarQuantizationConfig(
                    type=rest.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE_VECTOR_NAME: rest.VectorParams(
                    size=self._dimensions,
                    distance=rest.Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "bm25": rest.SparseVectorParams(
                    modifier=rest.Modifier.IDF,
                ),
                "splade": rest.SparseVectorParams(
                    modifier=rest.Modifier.IDF,
                ),
            },
            hnsw_config=hnsw_config,
            quantization_config=quantization_config,
            on_disk_payload=True,
        )

        if not self._is_local:
            for field_name, schema_type in [
                ("namespace", rest.PayloadSchemaType.KEYWORD),
                ("corpus_id", rest.PayloadSchemaType.KEYWORD),
                ("document_id", rest.PayloadSchemaType.KEYWORD),
                ("document_key", rest.PayloadSchemaType.KEYWORD),
                ("content_sha256", rest.PayloadSchemaType.KEYWORD),
                ("content_type", rest.PayloadSchemaType.KEYWORD),
                ("source_type", rest.PayloadSchemaType.KEYWORD),
            ]:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )

        logger.info(
            "Created Qdrant collection %s (dims=%d, quantization=%s, hnsw_ef=%d)",
            self._collection,
            self._dimensions,
            "INT8" if self._quantization_enabled else "none",
            100,
        )

    async def check_health(self) -> dict[str, object]:
        """Verify Qdrant is reachable and the collection exists."""
        health: dict[str, object] = {
            "healthy": False,
            "collection": self._collection,
            "dimensions": self._dimensions,
        }

        try:
            info = await self._client.get_collection(collection_name=self._collection)
            health["healthy"] = True
            health["points_count"] = info.points_count
            health["status"] = (
                info.status.value if hasattr(info.status, "value") else str(info.status)
            )

            if info.optimizer_status is not None:
                optimizer_status = info.optimizer_status
                optimizer_ok: bool | None = None
                if hasattr(optimizer_status, "ok"):
                    optimizer_ok = bool(getattr(optimizer_status, "ok"))
                elif hasattr(optimizer_status, "status"):
                    raw_status = getattr(optimizer_status, "status")
                    status_text = (
                        raw_status.value
                        if hasattr(raw_status, "value")
                        else str(raw_status)
                    ).lower()
                    optimizer_ok = status_text in {"ok", "green", "healthy"}
                if optimizer_ok is not None:
                    health["optimizer_ok"] = optimizer_ok

            health["write_latency_p50"] = self._latency.p50
            health["write_latency_p95"] = self._latency.p95
            health["write_latency_samples"] = self._latency.sample_count

        except Exception as exc:
            health["error"] = "%s: %s" % (type(exc).__name__, exc)
            logger.warning(
                "Qdrant health check failed for collection %s: %s",
                self._collection,
                exc,
            )

        return health

    async def _assert_collection_compatible(self) -> None:
        info = await self._client.get_collection(collection_name=self._collection)
        actual_dimensions = extract_dense_vector_size(info)
        if actual_dimensions is not None and actual_dimensions != self._dimensions:
            raise ValueError(
                "Existing collection %s uses %d dimensions, but the current embedding provider uses %d. "
                "Use a different collection name or reindex with a matching embedding configuration."
                % (self._collection, actual_dimensions, self._dimensions)
            )

        sparse_vector_names = extract_sparse_vector_names(info)
        if sparse_vector_names is None:
            return
        if "bm25" not in sparse_vector_names:
            raise ValueError(
                "Existing collection %s is missing the required sparse vector channel 'bm25'."
                % self._collection
            )
        self._available_sparse_vector_names = frozenset(sparse_vector_names)

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        """Upsert points with adaptive batch splitting and concurrency control."""
        if not points:
            return
        await self.ensure_collection()

        qdrant_points = [self._build_point(point) for point in points]
        batches = split_into_batches(qdrant_points, self._max_batch_size)

        for batch in batches:
            async with self._write_sem:
                await self._upsert_with_fallback(batch, split_depth=0)

    async def _upsert_with_fallback(
        self,
        points: list[rest.PointStruct],
        split_depth: int,
    ) -> None:
        await upsert_with_fallback(
            client=self._client,
            collection_name=self._collection,
            dimensions=self._dimensions,
            latency=self._latency,
            max_batch_size=self._max_batch_size,
            points=points,
            split_depth=split_depth,
        )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Hybrid search using Qdrant-native dense+sparse RRF fusion."""
        await self.ensure_collection()
        namespace = query.namespace.strip()
        if not namespace:
            raise ValueError("namespace is required for search")

        must_conditions: list[object] = [
            rest.FieldCondition(
                key="namespace",
                match=rest.MatchValue(value=namespace),
            ),
        ]
        if query.corpus_ids:
            must_conditions.append(
                rest.FieldCondition(
                    key="corpus_id",
                    match=rest.MatchAny(any=query.corpus_ids),
                ),
            )
        if query.content_types:
            must_conditions.append(
                rest.FieldCondition(
                    key="content_type",
                    match=rest.MatchAny(any=query.content_types),
                ),
            )
        if query.document_ids:
            must_conditions.append(
                rest.FieldCondition(
                    key="document_id",
                    match=rest.MatchAny(any=query.document_ids),
                ),
            )

        qdrant_filter = _build_filter(must_conditions)
        prefetch = self._build_prefetches(query=query, qdrant_filter=qdrant_filter)
        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),
            limit=query.limit,
            with_payload=True,
        )

        return [self._point_to_result(point) for point in response.points]

    def _build_point(self, point: VectorPoint) -> rest.PointStruct:
        return build_point(
            point,
            available_sparse_vector_names=self._available_sparse_vector_names,
        )

    def _build_prefetches(
        self,
        *,
        query: SearchQuery,
        qdrant_filter: rest.Filter,
    ) -> list[rest.Prefetch]:
        return build_prefetches(
            query=query,
            qdrant_filter=qdrant_filter,
            available_sparse_vector_names=self._available_sparse_vector_names,
        )

    @staticmethod
    def _point_to_result(point: rest.ScoredPoint) -> SearchResult:
        return point_to_result(point)

    async def delete(self, filter: DeleteFilter) -> None:
        """Delete points matching the filter."""
        await self.ensure_collection()
        namespace = (filter.namespace or "").strip()
        if not namespace:
            msg = "namespace is required for delete"
            raise ValueError(msg)

        must_conditions: list[object] = [
            rest.FieldCondition(
                key="namespace",
                match=rest.MatchValue(value=namespace),
            ),
        ]
        if filter.corpus_id:
            must_conditions.append(
                rest.FieldCondition(
                    key="corpus_id",
                    match=rest.MatchValue(value=filter.corpus_id),
                ),
            )
        if filter.document_id:
            must_conditions.append(
                rest.FieldCondition(
                    key="document_id",
                    match=rest.MatchValue(value=filter.document_id),
                ),
            )
        if not must_conditions:
            msg = "At least one filter field required for delete"
            raise ValueError(msg)

        await self._client.delete(
            collection_name=self._collection,
            points_selector=rest.FilterSelector(
                filter=_build_filter(must_conditions),
            ),
        )

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        await self.ensure_collection()
        await self._client.delete(
            collection_name=self._collection,
            points_selector=rest.PointIdsList(points=cast(Any, list(point_ids))),
        )

    async def get_document_record(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        await self.ensure_collection()
        namespace_scoped = namespace.strip()
        if not namespace_scoped:
            raise ValueError("namespace is required for get_document_record")
        corpus_scoped = corpus_id.strip()
        if not corpus_scoped:
            raise ValueError("corpus_id is required for get_document_record")
        if document_id is None and document_key is None:
            raise ValueError("document_id or document_key is required for get_document_record")

        # Shape: [FieldCondition("namespace"="team-space"), FieldCondition("corpus_id"="help-center")]
        must_conditions: list[object] = [
            rest.FieldCondition(
                key="namespace",
                match=rest.MatchValue(value=namespace_scoped),
            ),
            rest.FieldCondition(
                key="corpus_id",
                match=rest.MatchValue(value=corpus_scoped),
            ),
        ]
        if document_id is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="document_id",
                    match=rest.MatchValue(value=document_id),
                )
            )
        if document_key is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="document_key",
                    match=rest.MatchValue(value=document_key),
                )
            )

        records, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=_build_filter(must_conditions),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None

        payload = records[0].payload or {}
        chunk_count = await self._client.count(
            collection_name=self._collection,
            count_filter=_build_filter(
                [
                    rest.FieldCondition(
                        key="namespace",
                        match=rest.MatchValue(value=namespace_scoped),
                    ),
                    rest.FieldCondition(
                        key="corpus_id",
                        match=rest.MatchValue(value=corpus_scoped),
                    ),
                    rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(
                            value=str(payload.get("document_id") or document_id or "")
                        ),
                    ),
                ]
            ),
            exact=True,
        )
        return StoredDocumentRecord(
            document_id=str(payload.get("document_id") or document_id or ""),
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_key=(
                str(payload["document_key"])
                if payload.get("document_key") is not None
                else None
            ),
            content_sha256=(
                str(payload["content_sha256"])
                if payload.get("content_sha256") is not None
                else None
            ),
            chunk_count=int(chunk_count.count or 0),
        )


def _build_filter(must_conditions: Sequence[object]) -> rest.Filter:
    return rest.Filter(must=cast(Any, list(must_conditions)))
