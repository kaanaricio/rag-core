from __future__ import annotations

import uuid

from rag_core.search.types import ContentType
from rag_core.search.types import SparseVector, VectorPoint

from .indexer_embeddings import PreparedIndexData
from .indexer_models import IndexRequest


def build_points(
    *,
    req: IndexRequest,
    namespace: str,
    prepared: PreparedIndexData,
) -> tuple[list[VectorPoint], list[str]]:
    section_lookup = _build_section_lookup(req.section_mappings)
    points: list[VectorPoint] = []
    point_ids: list[str] = []

    for index, chunk in enumerate(prepared.chunks):
        point_id = make_point_id(
            namespace=namespace,
            corpus_id=req.corpus_id,
            document_id=req.document_id,
            chunk_index=index,
        )
        sparse_channels = prepared.sparse_channels[index]
        primary_sparse = _resolve_primary_sparse_vector(sparse_channels, index)
        payload = _build_payload(
            req=req,
            namespace=namespace,
            chunk_index=index,
            chunk_text=chunk.text,
            chunk_token_count=chunk.token_count,
            payload_text=prepared.payload_texts[index],
            content_type=prepared.content_type,
            section_info=section_lookup.get(index),
        )
        points.append(
            VectorPoint(
                id=point_id,
                dense_vector=prepared.dense_vectors[index],
                sparse_vector=primary_sparse,
                sparse_vectors=dict(sparse_channels),
                payload=payload,
            )
        )
        point_ids.append(point_id)

    return points, point_ids


def _build_payload(
    *,
    req: IndexRequest,
    namespace: str,
    chunk_index: int,
    chunk_text: str,
    chunk_token_count: int,
    payload_text: str,
    content_type: ContentType,
    section_info: dict[str, object] | None,
) -> dict[str, object]:
    chunk_word_count = len(chunk_text.split())
    chunk_token_estimate = int(chunk_token_count or max(1, round(chunk_word_count * 1.3)))
    document_path = req.document_path or req.path
    chunker_strategy = req.chunker_strategy or (
        "prechunked" if req.pre_chunked_texts else "content_chunker"
    )

    payload: dict[str, object] = {
        "namespace": namespace,
        "corpus_id": req.corpus_id,
        "document_id": req.document_id,
        "document_key": req.document_key,
        "content_sha256": req.content_sha256,
        "content_type": content_type,
        "chunk_index": chunk_index,
        "text": payload_text,
        "title": _resolve_display_title(req),
        "source_type": req.source_type,
        "mime_type": req.mime_type,
        "document_path": document_path,
        "chunk_word_count": chunk_word_count,
        "chunk_token_estimate": chunk_token_estimate,
        "chunker_strategy": chunker_strategy,
        "result_type": "text",
    }
    if req.embedding_model:
        payload["embedding_model"] = req.embedding_model
    if not section_info:
        return payload

    payload["section_id"] = section_info.get("section_id")
    payload["section_path"] = section_info.get("section_path")
    payload["section_title"] = section_info.get("section_title")
    for key in (
        "result_type",
        "figure_id",
        "image_url",
        "thumbnail_url",
        "figure_caption",
        "figure_bbox",
        "page_number",
        "bbox",
        "page_width",
        "page_height",
        "slide_number",
        "paragraph_index",
        "sheet_name",
        "row_range",
        "is_full_page",
        "anchor_chunk_index",
    ):
        value = section_info.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _resolve_display_title(req: IndexRequest) -> str:
    raw_title = (req.extra_fields or {}).get("title")
    if raw_title is None:
        return req.filename
    title = str(raw_title).strip()
    return title or req.filename


def _resolve_primary_sparse_vector(
    sparse_channels: dict[str, SparseVector],
    chunk_index: int,
) -> SparseVector:
    primary_sparse = sparse_channels.get("bm25")
    if primary_sparse is None and sparse_channels:
        primary_sparse = next(iter(sparse_channels.values()))
    if primary_sparse is None:
        raise ValueError("Sparse embedding is missing for chunk index %d" % chunk_index)
    return primary_sparse


def make_point_id(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_index: int,
) -> str:
    raw = f"{namespace.strip()}::{corpus_id.strip()}::{document_id}:chunk:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _build_section_lookup(
    mappings: list[dict[str, object]] | None,
) -> dict[int, dict[str, object]]:
    if not mappings:
        return {}

    section_lookup: dict[int, dict[str, object]] = {}
    for mapping in mappings:
        raw_index = mapping.get("chunk_index")
        if not isinstance(raw_index, int):
            continue
        section_path = mapping.get("section_path")
        section_title = mapping.get("section_title")
        if section_title is None and isinstance(section_path, str) and section_path.strip():
            section_title = section_path.split(">")[-1].strip()
        section_lookup[raw_index] = {
            **mapping,
            "section_title": section_title,
        }
    return section_lookup
