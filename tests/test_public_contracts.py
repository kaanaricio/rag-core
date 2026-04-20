import asyncio

import rag_core
from rag_core import (
    CorpusManifestEntry,
    IngestedDocument,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
    RAGCore,
    RAGCoreConfig,
    SearchResult,
)

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, RecordingVectorStore, make_search_result


def test_root_exports_are_available() -> None:
    assert CorpusManifestEntry.__name__ == "CorpusManifestEntry"
    assert RAGCore.__name__ == "RAGCore"
    assert RAGCoreConfig.__name__ == "RAGCoreConfig"
    assert ParsedDocument.__name__ == "ParsedDocument"
    assert IngestedDocument.__name__ == "IngestedDocument"
    assert PreparedChunk.__name__ == "PreparedChunk"
    assert PreparedDocument.__name__ == "PreparedDocument"
    assert "SearchResult" in rag_core.__all__


def test_rag_core_search_returns_public_hits() -> None:
    asyncio.run(_run_rag_core_search_returns_public_hits())


async def _run_rag_core_search_returns_public_hits() -> None:
    store = RecordingVectorStore(
        search_results=[
            make_search_result(
                id="hit-1",
                text="fox result",
                score=0.88,
                document_id="doc-7",
                corpus_id="corpus-a",
                metadata={"team": "search"},
            )
        ]
    )
    core = RAGCore(
        RAGCoreConfig(
            qdrant_location=":memory:",
            qdrant_collection="rag_core_public_contracts",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        hits = await core.search(
            query="fox query",
            namespace="team-space",
            corpus_ids=["corpus-a"],
            limit=3,
            document_ids=["doc-7"],
            rerank=False,
        )
    finally:
        await core.close()

    assert [type(hit) for hit in hits] == [SearchResult]
    assert hits[0].document_id == "doc-7"
    assert hits[0].metadata == {"team": "search"}

    query = store.search_calls[0]
    assert query.namespace == "team-space"
    assert query.corpus_ids == ["corpus-a"]
    assert query.document_ids == ["doc-7"]
