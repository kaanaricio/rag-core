import asyncio

from rag_core import RAGCore, RAGCoreConfig
from rag_core.search.types import SparseVector


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self._vocabulary = ("fox", "rag", "smoke", "tests")

    @property
    def dimensions(self) -> int:
        return len(self._vocabulary)

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        # Shape: [fox_hits, rag_hits, smoke_hits, tests_hits]
        return [float(lowered.count(term)) for term in self._vocabulary]


class FakeSparseEmbedder:
    def __init__(self) -> None:
        self._vocabulary = {"fox": 1, "rag": 2, "smoke": 3, "tests": 4}

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(text) for text in texts]

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        # Shape: [{"bm25": SparseVector(indices=[1, 3], values=[1.0, 1.0])}]
        return [{"bm25": self._embed(text)} for text in texts]

    def embed_query(self, query: str) -> SparseVector:
        return self._embed(query)

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        return {"bm25": self._embed(query)}

    def _embed(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            index = self._vocabulary.get(token.strip(".,!?"))
            if index is None:
                continue
            counts[index] = counts.get(index, 0.0) + 1.0
        return SparseVector(
            indices=list(counts.keys()),
            values=list(counts.values()),
        )


def test_local_ingest_and_search_smoke() -> None:
    asyncio.run(_run_local_ingest_and_search_smoke())


async def _run_local_ingest_and_search_smoke() -> None:
    embedding = FakeEmbeddingProvider()
    core = RAGCore(
        RAGCoreConfig(
            qdrant_location=":memory:",
            qdrant_collection="rag_core_test_smoke",
            embedding_dimensions=embedding.dimensions,
            contextualize=False,
        ),
        embedding_provider=embedding,
        sparse_embedder=FakeSparseEmbedder(),
    )

    try:
        ingested = await core.ingest_bytes(
            file_bytes=b"rag smoke tests keep the fox easy to find",
            filename="smoke.txt",
            mime_type="text/plain",
            namespace="test-space",
            corpus_id="test-corpus",
        )
        assert ingested.chunk_count > 0

        hits = await core.search(
            query="fox smoke",
            namespace="test-space",
            corpus_ids=["test-corpus"],
            limit=3,
            rerank=False,
        )
        assert hits
        assert hits[0].document_id == ingested.document_id
        assert "fox" in hits[0].text.lower()
    finally:
        await core.close()
