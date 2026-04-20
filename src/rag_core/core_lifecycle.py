from __future__ import annotations

import hashlib
import uuid

from rag_core.search.types import StoredDocumentRecord


def compute_content_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def resolve_document_key(
    *,
    filename: str,
    path: str | None,
    document_key: str | None,
) -> str:
    if document_key and document_key.strip():
        return document_key.strip()
    if path and path.strip():
        return path.strip()
    return filename.strip()


def resolve_document_id(
    *,
    namespace: str,
    corpus_id: str,
    document_key: str,
    document_id: str | None,
) -> str:
    if document_id and document_id.strip():
        return document_id.strip()
    raw = f"{namespace.strip()}::{corpus_id.strip()}::{document_key}"
    return f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, raw)}"


def resolve_ingest_state(
    existing: StoredDocumentRecord | None,
    *,
    content_sha256: str,
) -> tuple[str, bool]:
    if existing is None:
        return "created", True
    if existing.content_sha256 == content_sha256:
        return "unchanged", False
    return "replaced", True
