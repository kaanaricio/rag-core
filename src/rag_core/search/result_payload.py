from __future__ import annotations

from enum import Enum
from typing import Optional

from rag_core.search.types import SearchResult

_EXCLUDED_METADATA_KEYS = {
    "text",
    "content_type",
    "source_type",
    "document_id",
    "corpus_id",
    "document_key",
    "content_sha256",
    "title",
    "section_id",
    "section_title",
    "section_path",
    "document_path",
    "chunk_index",
    "chunk_word_count",
    "chunk_token_estimate",
    "embedding_model",
    "chunker_strategy",
    "result_type",
    "figure_id",
    "thumbnail_url",
    "figure_thumbnail_url",
    "namespace",
}


def payload_to_result(
    *,
    point_id: str,
    payload: dict[str, object],
    score: float,
) -> SearchResult:
    return SearchResult(
        id=point_id,
        text=str(payload["text"]),
        score=score,
        content_type=_required_str(payload, "content_type"),
        source_type=_optional_str(payload, "source_type") or "",
        document_id=_optional_str(payload, "document_id"),
        corpus_id=_optional_str(payload, "corpus_id"),
        document_key=_optional_str(payload, "document_key"),
        content_sha256=_optional_str(payload, "content_sha256"),
        title=_optional_str(payload, "title"),
        section_id=_optional_str(payload, "section_id"),
        section_title=_optional_str(payload, "section_title"),
        section_path=_optional_str(payload, "section_path"),
        document_path=_optional_str(payload, "document_path"),
        chunk_index=_optional_int(payload, "chunk_index"),
        chunk_word_count=_optional_int(payload, "chunk_word_count"),
        chunk_token_estimate=_optional_int(payload, "chunk_token_estimate"),
        embedding_model=_optional_str(payload, "embedding_model"),
        chunker_strategy=_optional_str(payload, "chunker_strategy"),
        result_type=_optional_str(payload, "result_type"),
        figure_id=_optional_str(payload, "figure_id"),
        figure_thumbnail_url=(
            _optional_str(payload, "thumbnail_url")
            or _optional_str(payload, "figure_thumbnail_url")
        ),
        metadata={key: value for key, value in payload.items() if key not in _EXCLUDED_METADATA_KEYS},
    )


def _required_str(payload: dict[str, object], key: str) -> str:
    return _stringify(payload[key])


def _optional_str(payload: dict[str, object], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    return _stringify(value)


def _optional_int(payload: dict[str, object], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    return int(value)


def _stringify(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
