from __future__ import annotations

from rag_core.core_lifecycle import (
    compute_content_sha256,
    resolve_document_id,
    resolve_document_key,
)
from rag_core.core_models import CorpusManifest, CorpusManifestEntry, IngestedDocument, PreparedDocument


def build_preview_document(
    *,
    file_bytes: bytes,
    prepared: PreparedDocument,
    namespace: str,
    corpus_id: str,
    document_id: str | None = None,
    document_key: str | None = None,
    metadata: dict[str, str] | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> IngestedDocument:
    resolved_document_key = resolve_document_key(
        filename=prepared.filename,
        path=prepared.path,
        document_key=document_key,
    )
    return IngestedDocument(
        document_id=resolve_document_id(
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=resolved_document_key,
            document_id=document_id,
        ),
        corpus_id=corpus_id,
        namespace=namespace,
        chunk_count=len(prepared.chunks),
        filename=prepared.filename,
        mime_type=prepared.mime_type,
        document_key=resolved_document_key,
        content_sha256=compute_content_sha256(file_bytes),
        ingest_state="preview",
        replaced_existing=False,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ocr=prepared.ocr,
        metadata={**prepared.metadata, **dict(metadata or {})},
    )


def build_manifest_entry(document: IngestedDocument) -> CorpusManifestEntry:
    return CorpusManifestEntry(
        document_id=document.document_id,
        namespace=document.namespace,
        corpus_id=document.corpus_id,
        document_key=document.document_key,
        content_sha256=document.content_sha256,
        filename=document.filename,
        mime_type=document.mime_type,
        chunk_count=document.chunk_count,
        parser=_optional_str(document.metadata.get("parser")),
        needs_ocr=bool(document.metadata.get("needs_ocr") or document.ocr.needed),
        metadata=dict(document.metadata),
    )


def build_corpus_manifest(
    *,
    namespace: str,
    corpus_id: str,
    collection_name: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    documents: list[IngestedDocument],
) -> CorpusManifest:
    ocr_document_count = sum(1 for document in documents if _ocr_used(document))
    ocr_page_count = sum(_ocr_page_count(document) for document in documents)
    entries = tuple(build_manifest_entry(document) for document in documents)
    return CorpusManifest(
        namespace=namespace,
        corpus_id=corpus_id,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        document_count=len(documents),
        chunk_count=sum(document.chunk_count for document in documents),
        source_document_ids=tuple(document.document_id for document in documents),
        ocr_document_count=ocr_document_count,
        ocr_page_count=ocr_page_count,
        entries=entries,
        documents=tuple(documents),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _ocr_used(document: IngestedDocument) -> bool:
    return bool(document.metadata.get("ocr_provider_used"))


def _ocr_page_count(document: IngestedDocument) -> int:
    explicit_count = document.metadata.get("ocr_pages_used_count")
    if isinstance(explicit_count, int) and explicit_count >= 0:
        return explicit_count
    raw_pages = document.metadata.get("ocr_pages_used")
    if not isinstance(raw_pages, list):
        return 0
    return sum(1 for page in raw_pages if isinstance(page, int) and page >= 0)
