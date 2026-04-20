from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import TypedDict

from rag_core.core import RAGCore, RAGCoreConfig
from rag_core.search.types import SparseVector

_DEMO_EMBEDDING_DIMENSIONS = 8
_DEMO_COLLECTION_PREFIX = "rag_core_examples"


class DemoHit(TypedDict):
    score: float
    title: str
    text: str


class DemoPayload(TypedDict):
    document_id: str
    chunk_count: int
    hits: list[DemoHit]


class DemoEmbeddingProvider:
    def __init__(self, *, dimensions: int = _DEMO_EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = max(1, dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "demo-dense-v1"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_dense_vector(text, dimensions=self._dimensions) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return _dense_vector(query, dimensions=self._dimensions)


class DemoSparseEmbedder:
    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        return [_sparse_vector(text) for text in texts]

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        return [{"bm25": vector} for vector in self.embed_texts(texts)]

    def embed_query(self, query: str) -> SparseVector:
        return _sparse_vector(query)

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        return {"bm25": self.embed_query(query)}


def build_demo_core(*, collection: str) -> RAGCore:
    return RAGCore(
        RAGCoreConfig(
            qdrant_location=":memory:",
            qdrant_collection=f"{_DEMO_COLLECTION_PREFIX}_{collection}",
            qdrant_dimension_aware_collection=False,
            embedding_provider="demo",
            embedding_model="demo-dense-v1",
            embedding_dimensions=_DEMO_EMBEDDING_DIMENSIONS,
            reranker_provider="none",
            contextualize=False,
        ),
        embedding_provider=DemoEmbeddingProvider(),
        sparse_embedder=DemoSparseEmbedder(),
    )


async def run_demo_app() -> DemoPayload:
    core = build_demo_core(collection="minimal_app")
    try:
        await core.ensure_ready()
        ingested = await core.ingest_bytes(
            file_bytes=b"Billing is due monthly and invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
        )
        hits = await core.search(
            query="How can I pay invoices?",
            namespace="acme",
            corpus_ids=["help-center"],
            limit=3,
            rerank=False,
        )
        return {
            "document_id": ingested.document_id,
            "chunk_count": ingested.chunk_count,
            "hits": [
                {
                    "score": hit.score,
                    "title": hit.title or hit.document_id or "unknown",
                    "text": hit.text,
                }
                for hit in hits
            ],
        }
    finally:
        await core.close()


def _dense_vector(text: str, *, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    for index, char in enumerate(text.lower()):
        bucket = (ord(char) + index) % dimensions
        values[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _sparse_vector(text: str) -> SparseVector:
    terms = Counter(token for token in text.lower().split() if token)
    if not terms:
        return SparseVector(indices=[0], values=[0.0])
    merged: dict[int, float] = {}
    for term, count in terms.items():
        index = zlib.adler32(term.encode("utf-8")) % 100_000
        merged[index] = merged.get(index, 0.0) + float(count)
    sorted_items = sorted(merged.items())
    return SparseVector(
        indices=[index for index, _ in sorted_items],
        values=[value for _, value in sorted_items],
    )
