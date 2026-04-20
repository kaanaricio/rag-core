from __future__ import annotations

from .vector_store_shared import _DENSE_VECTOR_NAME


def extract_dense_vector_size(collection_info: object) -> int | None:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)

    named_vectors = getattr(vectors, "_NamedVectorStruct__root", None)
    if isinstance(named_vectors, dict):
        dense = named_vectors.get(_DENSE_VECTOR_NAME)
        size = getattr(dense, "size", None)
        return int(size) if isinstance(size, int) else None

    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size

    if isinstance(vectors, dict):
        dense = vectors.get(_DENSE_VECTOR_NAME)
        size = getattr(dense, "size", None) if dense is not None else None
        if isinstance(size, int):
            return size

    return None


def extract_sparse_vector_names(collection_info: object) -> frozenset[str] | None:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)
    if sparse_vectors is None:
        return None

    named_vectors = getattr(sparse_vectors, "_NamedSparseVectorStruct__root", None)
    if isinstance(named_vectors, dict):
        return frozenset(str(name) for name in named_vectors if str(name))
    if isinstance(sparse_vectors, dict):
        return frozenset(str(name) for name in sparse_vectors if str(name))
    return None
