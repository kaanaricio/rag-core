from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IndexRequest:
    document_id: str
    corpus_id: str
    namespace: str
    text: str
    filename: str
    mime_type: str
    source_type: str
    document_key: str | None = None
    content_sha256: str | None = None
    existing_chunk_count: int | None = None
    path: Optional[str] = None
    document_path: Optional[str] = None
    section_mappings: Optional[list[dict[str, object]]] = None
    content_bytes: Optional[bytes] = None
    extra_fields: Optional[dict[str, str]] = None
    chunker_strategy: Optional[str] = None
    embedding_model: Optional[str] = None
    pre_chunked_texts: Optional[list[str]] = field(default=None)
    embedding_chunk_texts: Optional[list[str]] = field(default=None)


@dataclass(frozen=True)
class IndexResult:
    document_id: str
    chunk_count: int
    point_ids: list[str]
    point_payloads: list[dict[str, object]]
    document_key: str | None = None
    content_sha256: str | None = None
