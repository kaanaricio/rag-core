from __future__ import annotations

from typing import Any

from rag_core.documents.contextual_retriever import contextualize_chunks_for_embedding
from rag_core.documents.local_parse import parse_file_bytes
from rag_core.documents.ocr import OcrProvider, OcrRequest, OcrResult
from rag_core.search.chunking import chunk_content

from .core_models import OcrRoutingSignal, ParsedDocument, PreparedChunk, PreparedDocument


async def parse_document_bytes(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    path: str | None = None,
) -> ParsedDocument:
    markdown, metadata = await parse_file_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
    )
    return ParsedDocument(
        filename=filename,
        mime_type=mime_type,
        markdown=markdown,
        metadata=metadata,
        path=path,
    )


async def prepare_document_bytes(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    path: str | None,
    contextualize: bool,
    ocr_provider: OcrProvider | None,
) -> PreparedDocument:
    parsed = await parse_document_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        path=path,
    )
    if ocr_provider is not None and parsed.metadata.get("needs_ocr"):
        parsed = await apply_ocr(
            parsed=parsed,
            file_bytes=file_bytes,
            provider=ocr_provider,
        )

    chunk_results = chunk_content(parsed.markdown, mime_type=mime_type, filename=filename)
    chunk_texts = [chunk.text for chunk in chunk_results]
    embedding_texts = chunk_texts
    if contextualize and chunk_texts:
        embedding_texts = await contextualize_chunks_for_embedding(
            markdown=parsed.markdown,
            chunks=chunk_texts,
            filename=parsed.filename,
        )

    prepared_chunks = [
        PreparedChunk(
            chunk_index=index,
            text=chunk.text,
            embedding_text=embedding_texts[index],
            word_count=len(chunk.text.split()),
        )
        for index, chunk in enumerate(chunk_results)
    ]
    return PreparedDocument(
        filename=parsed.filename,
        mime_type=parsed.mime_type,
        markdown=parsed.markdown,
        chunks=prepared_chunks,
        metadata=parsed.metadata,
        path=parsed.path,
        ocr=build_ocr_signal(parsed.metadata),
    )


async def apply_ocr(
    *,
    parsed: ParsedDocument,
    file_bytes: bytes,
    provider: OcrProvider,
) -> ParsedDocument:
    page_indices = normalize_page_indices(parsed.metadata.get("ocr_page_indices"))
    ocr_result = await provider.extract_markdown(
        OcrRequest(
            file_bytes=file_bytes,
            filename=parsed.filename,
            mime_type=parsed.mime_type,
            page_indices=page_indices,
            existing_markdown=parsed.markdown,
            metadata=dict(parsed.metadata),
        )
    )
    merged_metadata = dict(parsed.metadata)
    merged_metadata.update(ocr_result.metadata)
    if ocr_result.provider_name:
        merged_metadata["ocr_provider"] = ocr_result.provider_name
    if ocr_result.model_name:
        merged_metadata["ocr_model"] = ocr_result.model_name
    ocr_pages_used = _resolve_ocr_pages_used(
        parsed_metadata=parsed.metadata,
        ocr_result=ocr_result,
        requested_page_indices=page_indices,
    )
    merged_metadata["ocr_pages_used"] = ocr_pages_used
    merged_metadata["ocr_pages_used_count"] = _resolve_ocr_page_count(
        parsed_metadata=parsed.metadata,
        ocr_result=ocr_result,
        ocr_pages_used=ocr_pages_used,
        requested_page_indices=page_indices,
    )
    merged_metadata["ocr_merge_mode"] = ocr_result.merge_mode
    if ocr_result.pages_processed:
        merged_metadata["ocr_page_indices"] = list(ocr_result.pages_processed)
    merged_metadata["ocr_provider_used"] = True
    merged_metadata["needs_ocr"] = False
    return ParsedDocument(
        filename=parsed.filename,
        mime_type=parsed.mime_type,
        markdown=merge_markdown(parsed.markdown, ocr_result),
        metadata=merged_metadata,
        path=parsed.path,
    )


def build_ocr_signal(metadata: dict[str, Any]) -> OcrRoutingSignal:
    return OcrRoutingSignal(
        needed=bool(metadata.get("needs_ocr")),
        page_indices=normalize_page_indices(metadata.get("ocr_page_indices")),
        confidence=coerce_float(metadata.get("confidence")),
        parser=coerce_str(metadata.get("parser")),
    )


def normalize_page_indices(raw_indices: Any) -> list[int]:
    if not isinstance(raw_indices, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        if not isinstance(raw_index, int) or raw_index < 0 or raw_index in seen:
            continue
        seen.add(raw_index)
        normalized.append(raw_index)
    return sorted(normalized)


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_ocr_pages_used(
    *,
    parsed_metadata: dict[str, Any],
    ocr_result: OcrResult,
    requested_page_indices: list[int],
) -> list[int]:
    processed_pages = normalize_page_indices(ocr_result.pages_processed)
    if processed_pages:
        return processed_pages
    # Shape: {"ocr_processed_entire_document": True}
    if bool(ocr_result.metadata.get("ocr_processed_entire_document")):
        page_count = _resolve_document_page_count(
            parsed_metadata=parsed_metadata,
            ocr_metadata=ocr_result.metadata,
        )
        if page_count is not None:
            return list(range(page_count))
        return []
    return list(requested_page_indices)


def _resolve_ocr_page_count(
    *,
    parsed_metadata: dict[str, Any],
    ocr_result: OcrResult,
    ocr_pages_used: list[int],
    requested_page_indices: list[int],
) -> int:
    if ocr_pages_used:
        return len(ocr_pages_used)
    explicit_count = coerce_int(ocr_result.metadata.get("ocr_pages_used_count"))
    if explicit_count is not None and explicit_count >= 0:
        return explicit_count
    if bool(ocr_result.metadata.get("ocr_processed_entire_document")):
        page_count = _resolve_document_page_count(
            parsed_metadata=parsed_metadata,
            ocr_metadata=ocr_result.metadata,
        )
        if page_count is not None:
            return page_count
        return 0
    return len(requested_page_indices)


def _resolve_document_page_count(
    *,
    parsed_metadata: dict[str, Any],
    ocr_metadata: dict[str, Any],
) -> int | None:
    for raw_value in (
        parsed_metadata.get("page_count"),
        ocr_metadata.get("page_count"),
        ocr_metadata.get("ocr_page_count"),
    ):
        page_count = coerce_int(raw_value)
        if page_count is not None and page_count >= 0:
            return page_count
    return None


def merge_markdown(base_markdown: str, ocr_result: OcrResult) -> str:
    ocr_markdown = ocr_result.markdown.strip()
    if not ocr_markdown:
        return base_markdown
    if ocr_result.merge_mode == "replace":
        return ocr_markdown
    base = base_markdown.strip()
    if not base:
        return ocr_markdown
    return f"{base}\n\n{ocr_markdown}"
