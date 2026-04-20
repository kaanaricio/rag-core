from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RAGCoreConfig:
    qdrant_url: str | None = None
    qdrant_location: str | None = None
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_core_chunks"
    qdrant_dimension_aware_collection: bool = True
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    reranker_provider: str = "none"
    reranker_model: str | None = None
    reranker_api_key: str | None = None
    contextualize: bool = True
    source_type: str = "file"
    enable_exact_match_sidecar: bool = False


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    mime_type: str
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    corpus_id: str
    namespace: str
    chunk_count: int
    filename: str
    mime_type: str
    document_key: str | None = None
    content_sha256: str | None = None
    ingest_state: str = "created"
    replaced_existing: bool = False
    collection_name: str | None = None
    embedding_model: str | None = None
    ocr: "OcrRoutingSignal" = field(default_factory=lambda: OcrRoutingSignal())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusManifestEntry:
    document_id: str
    namespace: str
    corpus_id: str
    document_key: str | None
    content_sha256: str | None
    filename: str
    mime_type: str
    chunk_count: int
    parser: str | None = None
    needs_ocr: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedChunk:
    chunk_index: int
    text: str
    embedding_text: str
    word_count: int


@dataclass(frozen=True)
class OcrRoutingSignal:
    needed: bool = False
    page_indices: list[int] = field(default_factory=list)
    confidence: float | None = None
    parser: str | None = None


@dataclass(frozen=True)
class PreparedDocument:
    filename: str
    mime_type: str
    markdown: str
    chunks: list[PreparedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    ocr: OcrRoutingSignal = field(default_factory=OcrRoutingSignal)


@dataclass(frozen=True)
class CorpusManifest:
    namespace: str
    corpus_id: str
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    document_count: int
    chunk_count: int
    source_document_ids: tuple[str, ...]
    ocr_document_count: int
    ocr_page_count: int
    documents: tuple[IngestedDocument, ...]
    entries: tuple[CorpusManifestEntry, ...] = ()
