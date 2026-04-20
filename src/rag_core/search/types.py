"""Core types for the search infrastructure."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

_DEFAULT_SPARSE_CHANNEL = "bm25"


class ContentType(str, enum.Enum):
    DOCUMENT = "document"
    CODE = "code"


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation (BM25 indices + values)."""

    indices: list[int]
    values: list[float]


def _merge_sparse_channels(
    primary: SparseVector,
    extra: dict[str, SparseVector] | None,
) -> dict[str, SparseVector]:
    merged: dict[str, SparseVector] = {_DEFAULT_SPARSE_CHANNEL: primary}
    if not extra:
        return merged
    for name, vector in extra.items():
        if not name:
            continue
        merged[str(name)] = vector
    return merged


@dataclass(frozen=True)
class ChunkResult:
    """Output of a chunking operation."""

    text: str
    start_index: int
    end_index: int
    token_count: int


@dataclass(frozen=True)
class TextualRepresentation:
    """Metadata header + content for a chunk, ready for embedding."""

    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class VectorPoint:
    """A point to upsert into the vector store."""

    id: str
    dense_vector: list[float]
    sparse_vector: SparseVector
    payload: dict[str, object]
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)

    def all_sparse_vectors(self) -> dict[str, SparseVector]:
        """Return sparse vectors keyed by channel name (always includes bm25)."""
        return _merge_sparse_channels(self.sparse_vector, self.sparse_vectors)


@dataclass(frozen=True)
class SearchResult:
    """A single search result from any source.

    Example: SearchResult(id="uuid5-hex", text="# Metadata\\n...\\n# Content\\n...",
             score=0.87, content_type="document", source_type="file",
             document_id="doc_123", corpus_id="help_center",
             document_key="docs/report.pdf", title="Q1 Report",
             chunk_index=3, section_title="Introduction")
    """

    id: str
    text: str
    score: float
    content_type: str
    source_type: str
    document_id: Optional[str] = None
    corpus_id: Optional[str] = None
    document_key: Optional[str] = None
    content_sha256: Optional[str] = None
    title: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    document_path: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_word_count: Optional[int] = None
    chunk_token_estimate: Optional[int] = None
    embedding_model: Optional[str] = None
    chunker_strategy: Optional[str] = None
    result_type: Optional[str] = None
    figure_id: Optional[str] = None
    figure_thumbnail_url: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredDocumentRecord:
    document_id: str
    namespace: str
    corpus_id: str
    document_key: str | None = None
    content_sha256: str | None = None
    chunk_count: int = 0


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking a document."""

    index: int
    score: float
    text: str


@dataclass
class SearchQuery:
    """Parameters for a search query."""

    dense_vector: list[float]
    sparse_vector: SparseVector
    namespace: str
    corpus_ids: list[str]
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    limit: int = 20

    def all_sparse_vectors(self) -> dict[str, SparseVector]:
        """Return query sparse vectors keyed by channel name (always includes bm25)."""
        return _merge_sparse_channels(self.sparse_vector, self.sparse_vectors)


@dataclass(frozen=True)
class SearchSidecarQuery:
    """Portable search request for optional lexical/exact sidecars."""

    query: str
    namespace: str
    corpus_ids: list[str]
    limit: int = 20
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None


@dataclass
class DeleteFilter:
    """Filter for deleting points from the vector store."""

    namespace: Optional[str] = None
    corpus_id: Optional[str] = None
    document_id: Optional[str] = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense embedding providers (OpenAI, Voyage, etc.)."""

    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Protocol for sparse embedding (BM25 via FastEmbed)."""

    def embed_texts(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, query: str) -> SparseVector: ...

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]: ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Protocol for reranking providers (Cohere, Jina, etc.)."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]: ...


@runtime_checkable
class SearchSidecar(Protocol):
    """Protocol for optional lexical/exact-match sidecars."""

    def upsert_records(self, records: Sequence[object]) -> None: ...

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None: ...

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store backends (Qdrant, etc.)."""

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def search(self, query: SearchQuery) -> list[SearchResult]: ...

    async def delete(self, filter: DeleteFilter) -> None: ...

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None: ...

    async def ensure_collection(self) -> None: ...

    async def check_health(self) -> dict[str, object]: ...

    async def get_document_record(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None: ...

    async def close(self) -> None: ...
