"""Search-time helpers for Qdrant vector store conversion and filtering."""

from __future__ import annotations

from qdrant_client import models as rest

from rag_core.search.result_payload import payload_to_result
from rag_core.search.types import SearchQuery, SearchResult, VectorPoint

from .vector_store_shared import _DENSE_VECTOR_NAME, _KNOWN_SPARSE_VECTOR_NAMES, _PREFETCH_LIMIT


def build_point(
    point: VectorPoint,
    available_sparse_vector_names: frozenset[str] | set[str] = _KNOWN_SPARSE_VECTOR_NAMES,
) -> rest.PointStruct:
    """Convert a ``VectorPoint`` to a Qdrant ``PointStruct``."""
    sparse_dict = {
        name: rest.SparseVector(indices=vector.indices, values=vector.values)
        for name, vector in point.all_sparse_vectors().items()
        if name in available_sparse_vector_names
    }
    return rest.PointStruct(
        id=point.id,
        vector={_DENSE_VECTOR_NAME: point.dense_vector, **sparse_dict},
        payload=point.payload,
    )


def build_prefetches(
    *,
    query: SearchQuery,
    qdrant_filter: rest.Filter,
    available_sparse_vector_names: frozenset[str] | set[str] = _KNOWN_SPARSE_VECTOR_NAMES,
) -> list[rest.Prefetch]:
    """Build dense and sparse prefetches for hybrid search."""
    prefetch: list[rest.Prefetch] = [
        rest.Prefetch(
            query=query.dense_vector,
            using=_DENSE_VECTOR_NAME,
            limit=_PREFETCH_LIMIT,
            filter=qdrant_filter,
        ),
    ]

    for name, sparse_vector in query.all_sparse_vectors().items():
        if name not in available_sparse_vector_names:
            continue
        prefetch.append(
            rest.Prefetch(
                query=rest.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using=name,
                limit=_PREFETCH_LIMIT,
                filter=qdrant_filter,
            ),
        )

    return prefetch


def point_to_result(point: rest.ScoredPoint) -> SearchResult:
    """Convert a Qdrant ``ScoredPoint`` into our search result type."""
    return payload_to_result(
        point_id=str(point.id),
        payload=point.payload or {},
        score=point.score or 0.0,
    )
