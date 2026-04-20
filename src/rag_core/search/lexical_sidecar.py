"""Portable lexical sidecar for exact or trigram-style matching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from rag_core.search.types import SearchResult, SearchSidecar, SearchSidecarQuery


@dataclass(frozen=True)
class LexicalSidecarRecord:
    """Portable record shape for sidecar lookups."""

    namespace: str
    result: SearchResult


class PortableLexicalSidecar(SearchSidecar):
    """Small in-memory sidecar for exact and trigram-style retrieval."""

    def __init__(
        self,
        records: list[LexicalSidecarRecord],
        *,
        trigram_threshold: float = 0.35,
    ) -> None:
        self._records = list(records)
        self._trigram_threshold = trigram_threshold

    def upsert_records(self, records: Sequence[object]) -> None:
        existing = {
            (record.namespace, record.result.id): record
            for record in self._records
        }
        for record in records:
            if not isinstance(record, LexicalSidecarRecord):
                continue
            existing[(record.namespace, record.result.id)] = record
        self._records = list(existing.values())

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None:
        self._records = [
            record
            for record in self._records
            if not (
                record.namespace == namespace
                and record.result.document_id == document_id
                and (corpus_id is None or record.result.corpus_id == corpus_id)
            )
        ]

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        needle = _normalize(query.query)
        if not needle:
            return []

        matches: list[tuple[float, SearchResult]] = []
        for record in self._records:
            if not _matches_filters(record, query):
                continue

            searchable_fields = _searchable_fields(record.result)
            best = _best_match(needle, searchable_fields, self._trigram_threshold)
            if best is None:
                continue

            score, strategy, field_name, matched_value = best
            matches.append(
                (
                    score,
                    _annotate_result(
                        record.result,
                        score=score,
                        strategy=strategy,
                        field_name=field_name,
                        matched_value=matched_value,
                    ),
                )
            )

        matches.sort(key=lambda item: (item[0], item[1].score), reverse=True)
        return [result for _, result in matches[: query.limit]]


def _matches_filters(record: LexicalSidecarRecord, query: SearchSidecarQuery) -> bool:
    result = record.result
    if record.namespace != query.namespace:
        return False
    if query.corpus_ids and result.corpus_id not in query.corpus_ids:
        return False
    if query.content_types and result.content_type not in query.content_types:
        return False
    if query.document_ids and result.document_id not in query.document_ids:
        return False
    return True


def _searchable_fields(result: SearchResult) -> list[tuple[str, str]]:
    # Shape: [("title", "Fox Query"), ("section_title", "Overview"), ("text", "...")]
    raw_values = [
        ("title", result.title),
        ("section_title", result.section_title),
        ("section_path", result.section_path),
        ("document_path", result.document_path),
        ("text", result.text),
    ]
    return [(name, value) for name, value in raw_values if value]


def _best_match(
    needle: str,
    searchable_fields: list[tuple[str, str]],
    trigram_threshold: float,
) -> tuple[float, str, str, str] | None:
    best: tuple[float, str, str, str] | None = None
    for field_name, raw_value in searchable_fields:
        candidate = _normalize(raw_value)
        if not candidate:
            continue

        if candidate == needle:
            return (1.0, "exact", field_name, raw_value)
        if f" {needle} " in f" {candidate} ":
            score = 0.9
            if best is None or score > best[0]:
                best = (score, "exact", field_name, raw_value)
            continue

        score = _trigram_score(needle, candidate)
        if score < trigram_threshold:
            continue
        if best is None or score > best[0]:
            best = (score, "trigram", field_name, raw_value)
    return best


def _annotate_result(
    result: SearchResult,
    *,
    score: float,
    strategy: str,
    field_name: str,
    matched_value: str,
) -> SearchResult:
    metadata = dict(result.metadata)
    metadata["search_sidecar"] = {
        "score": score,
        "field": field_name,
        "strategy": strategy,
        "matched_value": matched_value,
    }
    return replace(result, score=score, metadata=metadata)


def _normalize(value: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _trigram_score(left: str, right: str) -> float:
    left_trigrams = _trigrams(left)
    right_trigrams = _trigrams(right)
    if not left_trigrams or not right_trigrams:
        return 0.0
    overlap = len(left_trigrams & right_trigrams)
    total = len(left_trigrams | right_trigrams)
    if total == 0:
        return 0.0
    return overlap / total


def _trigrams(value: str) -> set[str]:
    padded = f"  {value}  "
    if len(padded) < 3:
        return set()
    return {padded[index : index + 3] for index in range(len(padded) - 2)}
