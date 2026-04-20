from __future__ import annotations

from typing import Sequence

from rag_core.search.types import (
    DeleteFilter,
    RerankResult,
    SearchQuery,
    SearchResult,
    SearchSidecarQuery,
    SparseVector,
    StoredDocumentRecord,
    VectorPoint,
)


class FakeEmbeddingProvider:
    def __init__(self, vocabulary: tuple[str, ...] = ("original", "context", "fox", "query")) -> None:
        self._vocabulary = vocabulary
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    @property
    def dimensions(self) -> int:
        return len(self._vocabulary)

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(term)) for term in self._vocabulary]


class FakeSparseEmbedder:
    def __init__(
        self,
        *,
        raise_on_multi: bool = False,
        include_extra_channel: bool = True,
        empty_query_multi: bool = False,
    ) -> None:
        self._vocabulary = {"original": 1, "context": 2, "fox": 3, "query": 4}
        self.raise_on_multi = raise_on_multi
        self.include_extra_channel = include_extra_channel
        self.empty_query_multi = empty_query_multi
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []
        self.embed_texts_multi_calls: list[list[str]] = []
        self.embed_query_multi_calls: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        self.embed_texts_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        self.embed_texts_multi_calls.append(list(texts))
        if self.raise_on_multi:
            raise RuntimeError("multi failed")
        merged = [{"bm25": self._embed(text)} for text in texts]
        if not self.include_extra_channel:
            return merged
        for item, text in zip(merged, texts, strict=True):
            item["splade"] = self._embed(f"context {text}")
        return merged

    def embed_query(self, query: str) -> SparseVector:
        self.embed_query_calls.append(query)
        return self._embed(query)

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        self.embed_query_multi_calls.append(query)
        if self.empty_query_multi:
            return {}
        merged = {"bm25": self._embed(query)}
        if self.include_extra_channel:
            merged["splade"] = self._embed(f"context {query}")
        return merged

    def _embed(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            index = self._vocabulary.get(token.strip(".,!?"))
            if index is None:
                continue
            counts[index] = counts.get(index, 0.0) + 1.0
        return SparseVector(indices=list(counts.keys()), values=list(counts.values()))


class FakeSparseEmbedderNoMulti:
    def __init__(self) -> None:
        self._delegate = FakeSparseEmbedder(include_extra_channel=False)

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        return self._delegate.embed_texts(texts)

    def embed_query(self, query: str) -> SparseVector:
        return self._delegate.embed_query(query)

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        return self._delegate.embed_query_multi(query)


class RecordingVectorStore:
    def __init__(
        self,
        *,
        search_results: list[SearchResult] | None = None,
        health: dict[str, object] | None = None,
        document_records: dict[tuple[str, str, str], StoredDocumentRecord] | None = None,
    ) -> None:
        self.search_results = list(search_results or [])
        self.health = health or {"ok": True}
        self.document_records = dict(document_records or {})
        self.operations: list[str] = []
        self.upsert_calls: list[list[VectorPoint]] = []
        self.search_calls: list[SearchQuery] = []
        self.delete_calls: list[DeleteFilter] = []
        self.delete_point_ids_calls: list[list[str]] = []
        self.get_document_record_calls: list[tuple[str, str, str | None, str | None]] = []
        self.ensure_collection_calls = 0
        self.close_calls = 0

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        self.operations.append("upsert")
        self.upsert_calls.append(list(points))
        for key, chunk_count in _count_document_chunks(points).items():
            namespace, corpus_id, document_id = key
            sample_point = next(
                point
                for point in points
                if (
                    str(point.payload.get("namespace") or ""),
                    str(point.payload.get("corpus_id") or ""),
                    str(point.payload.get("document_id") or ""),
                )
                == key
            )
            self.document_records[key] = StoredDocumentRecord(
                document_id=document_id,
                namespace=namespace,
                corpus_id=corpus_id,
                document_key=(
                    str(sample_point.payload["document_key"])
                    if sample_point.payload.get("document_key") is not None
                    else None
                ),
                content_sha256=(
                    str(sample_point.payload["content_sha256"])
                    if sample_point.payload.get("content_sha256") is not None
                    else None
                ),
                chunk_count=chunk_count,
            )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        self.operations.append("search")
        self.search_calls.append(query)
        return list(self.search_results)

    async def delete(self, filter: DeleteFilter) -> None:
        self.operations.append("delete")
        self.delete_calls.append(filter)
        self.document_records = {
            key: record
            for key, record in self.document_records.items()
            if not (
                (filter.namespace is None or record.namespace == filter.namespace)
                and (filter.corpus_id is None or record.corpus_id == filter.corpus_id)
                and (filter.document_id is None or record.document_id == filter.document_id)
            )
        }

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        self.operations.append("delete_point_ids")
        self.delete_point_ids_calls.append(list(point_ids))

    async def ensure_collection(self) -> None:
        self.operations.append("ensure_collection")
        self.ensure_collection_calls += 1

    async def check_health(self) -> dict[str, object]:
        self.operations.append("check_health")
        return dict(self.health)

    async def get_document_record(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        self.operations.append("get_document_record")
        self.get_document_record_calls.append((namespace, corpus_id, document_id, document_key))
        if document_id is not None:
            return self.document_records.get((namespace, corpus_id, document_id))
        for record in self.document_records.values():
            if (
                record.namespace == namespace
                and record.corpus_id == corpus_id
                and record.document_key == document_key
            ):
                return record
        return None

    async def close(self) -> None:
        self.operations.append("close")
        self.close_calls += 1


class FakeReranker:
    def __init__(
        self,
        *,
        results: list[RerankResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results or [])
        self._error = error
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents), top_k))
        if self._error is not None:
            raise self._error
        return list(self._results)


class FakeSearchSidecar:
    def __init__(
        self,
        *,
        results: list[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results or [])
        self._error = error
        self.calls: list[SearchSidecarQuery] = []
        self.upserted: list[object] = []
        self.deleted: list[tuple[str, str]] = []

    def upsert_records(self, records: Sequence[object]) -> None:
        self.upserted.extend(records)

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None:
        self.deleted.append((namespace, document_id))

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return list(self._results)


def make_search_result(
    *,
    id: str = "result-1",
    text: str = "fox query context",
    score: float = 0.9,
    content_type: str = "document",
    source_type: str = "file",
    document_id: str | None = "doc-1",
    corpus_id: str | None = "corpus-1",
    document_key: str | None = None,
    content_sha256: str | None = None,
    title: str | None = "Doc 1",
    section_id: str | None = None,
    section_title: str | None = None,
    section_path: str | None = None,
    document_path: str | None = None,
    chunk_index: int | None = 0,
    chunk_word_count: int | None = None,
    chunk_token_estimate: int | None = None,
    embedding_model: str | None = None,
    chunker_strategy: str | None = None,
    result_type: str | None = None,
    figure_id: str | None = None,
    figure_thumbnail_url: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    return SearchResult(
        id=id,
        text=text,
        score=score,
        content_type=content_type,
        source_type=source_type,
        document_id=document_id,
        corpus_id=corpus_id,
        document_key=document_key,
        content_sha256=content_sha256,
        title=title,
        section_id=section_id,
        section_title=section_title,
        section_path=section_path,
        document_path=document_path,
        chunk_index=chunk_index,
        chunk_word_count=chunk_word_count,
        chunk_token_estimate=chunk_token_estimate,
        embedding_model=embedding_model,
        chunker_strategy=chunker_strategy,
        result_type=result_type,
        figure_id=figure_id,
        figure_thumbnail_url=figure_thumbnail_url,
        metadata=metadata or {"source": "fixture"},
    )


def _count_document_chunks(points: Sequence[VectorPoint]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for point in points:
        key = (
            str(point.payload.get("namespace") or ""),
            str(point.payload.get("corpus_id") or ""),
            str(point.payload.get("document_id") or ""),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts
