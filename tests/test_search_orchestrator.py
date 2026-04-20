import asyncio
from typing import cast

from rag_core.search.lexical_sidecar import LexicalSidecarRecord, PortableLexicalSidecar
from rag_core.search.searcher import SearchOrchestrator, SearchRequest
from rag_core.search.types import RerankResult, SparseVector

from tests.support import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSearchSidecar,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


def test_search_uses_precomputed_query_vectors() -> None:
    asyncio.run(_run_search_uses_precomputed_query_vectors())


async def _run_search_uses_precomputed_query_vectors() -> None:
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()
    store = RecordingVectorStore(search_results=[make_search_result()])
    orchestrator = SearchOrchestrator(
        embedding_provider=embedding,
        sparse_embedder=sparse,
        vector_store=store,
    )
    dense = [9.0, 8.0, 7.0, 6.0]
    sparse_vectors = {"bm25": SparseVector(indices=[1, 2], values=[1.0, 2.0])}

    results = await orchestrator.search(
        SearchRequest(
            query="unused",
            corpus_ids=["corpus-1"],
            namespace="space-1",
            query_vector=dense,
            query_sparse_vectors=sparse_vectors,
        )
    )

    assert len(results) == 1
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []
    assert store.search_calls[0].dense_vector == dense
    assert store.search_calls[0].sparse_vector == sparse_vectors["bm25"]


def test_search_uses_first_sparse_channel_when_bm25_missing() -> None:
    asyncio.run(_run_search_uses_first_sparse_channel_when_bm25_missing())


async def _run_search_uses_first_sparse_channel_when_bm25_missing() -> None:
    store = RecordingVectorStore()
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    splade = SparseVector(indices=[9], values=[3.0])

    await orchestrator.search(
        SearchRequest(
            query="query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
            query_vector=[1.0, 2.0, 3.0, 4.0],
            query_sparse_vectors={"splade": splade},
        )
    )

    assert store.search_calls[0].sparse_vector == splade


def test_search_raises_when_no_sparse_vector_is_available() -> None:
    asyncio.run(_run_search_raises_when_no_sparse_vector_is_available())


async def _run_search_raises_when_no_sparse_vector_is_available() -> None:
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(empty_query_multi=True),
        vector_store=RecordingVectorStore(),
    )

    try:
        await orchestrator.search(
            SearchRequest(
                query="query",
                corpus_ids=["corpus-1"],
                namespace="space-1",
                query_vector=[1.0, 2.0, 3.0, 4.0],
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "No sparse query vector generated"
    else:
        raise AssertionError("Expected RuntimeError when no sparse vector is available")


def test_search_reranks_results() -> None:
    asyncio.run(_run_search_reranks_results())


async def _run_search_reranks_results() -> None:
    doc_a = make_search_result(id="doc-a", text="fox alpha")
    doc_b = make_search_result(id="doc-b", text="query beta")
    reranker = FakeReranker(
        results=[
            RerankResult(index=1, score=0.95, text=doc_b.text),
            RerankResult(index=0, score=0.90, text=doc_a.text),
        ]
    )
    store = RecordingVectorStore(search_results=[doc_a, doc_b])

    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        reranker=reranker,
    )

    results = await orchestrator.search(
        SearchRequest(
            query="query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
            rerank=True,
        )
    )

    assert [result.id for result in results] == ["doc-b", "doc-a"]
    assert reranker.calls == [("query", [doc_a.text, doc_b.text], 20)]


def test_search_returns_original_order_when_rerank_fails() -> None:
    asyncio.run(_run_search_returns_original_order_when_rerank_fails())


async def _run_search_returns_original_order_when_rerank_fails() -> None:
    doc = make_search_result(id="doc-a", text="fox alpha")
    store = RecordingVectorStore(search_results=[doc])

    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        reranker=FakeReranker(error=RuntimeError("rerank failed")),
    )

    results = await orchestrator.search(
        SearchRequest(
            query="query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
            rerank=True,
        )
    )

    assert [result.id for result in results] == ["doc-a"]


def test_search_merges_sidecar_results_before_vector_results() -> None:
    asyncio.run(_run_search_merges_sidecar_results_before_vector_results())


async def _run_search_merges_sidecar_results_before_vector_results() -> None:
    semantic = make_search_result(id="doc-semantic", text="fox context")
    exact = make_search_result(
        id="doc-exact",
        text="fox query text",
        title="Fox Query",
        score=1.0,
    )
    store = RecordingVectorStore(search_results=[semantic])
    sidecar = FakeSearchSidecar(results=[exact])
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        sidecar=sidecar,
    )

    results = await orchestrator.search(
        SearchRequest(
            query="fox query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
        )
    )

    assert [result.id for result in results] == ["doc-exact", "doc-semantic"]
    assert sidecar.calls[0].namespace == "space-1"
    assert sidecar.calls[0].corpus_ids == ["corpus-1"]


def test_search_dedupes_sidecar_results_by_id() -> None:
    asyncio.run(_run_search_dedupes_sidecar_results_by_id())


async def _run_search_dedupes_sidecar_results_by_id() -> None:
    vector_hit = make_search_result(
        id="doc-1",
        text="semantic hit",
        score=0.6,
        document_key="/docs/guide.txt",
        content_sha256="hash-1",
        section_title="Overview",
    )
    sidecar_hit = make_search_result(id="doc-1", text="exact hit", score=0.0)
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(search_results=[vector_hit]),
        sidecar=FakeSearchSidecar(results=[sidecar_hit]),
    )

    results = await orchestrator.search(
        SearchRequest(
            query="fox query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
        )
    )

    assert [result.id for result in results] == ["doc-1"]
    assert results[0].text == "exact hit"
    assert results[0].score == 0.6
    assert results[0].document_key == "/docs/guide.txt"
    assert results[0].content_sha256 == "hash-1"
    assert results[0].section_title == "Overview"


def test_search_can_disable_sidecar_per_request() -> None:
    asyncio.run(_run_search_can_disable_sidecar_per_request())


async def _run_search_can_disable_sidecar_per_request() -> None:
    sidecar = FakeSearchSidecar(results=[make_search_result(id="doc-exact", score=1.0)])
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(search_results=[make_search_result(id="doc-store")]),
        sidecar=sidecar,
    )

    results = await orchestrator.search(
        SearchRequest(
            query="fox query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
            use_sidecar=False,
        )
    )

    assert [result.id for result in results] == ["doc-store"]
    assert sidecar.calls == []


def test_portable_lexical_sidecar_promotes_exact_and_trigram_matches() -> None:
    asyncio.run(_run_portable_lexical_sidecar_promotes_exact_and_trigram_matches())


async def _run_portable_lexical_sidecar_promotes_exact_and_trigram_matches() -> None:
    exact = make_search_result(id="doc-exact", title="Fox Query", text="semantic text")
    trigram = make_search_result(id="doc-trigram", title="Foks Queri", text="semantic text")
    sidecar = PortableLexicalSidecar(
        records=[
            LexicalSidecarRecord(namespace="space-1", result=trigram),
            LexicalSidecarRecord(namespace="space-1", result=exact),
        ]
    )
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
        sidecar=sidecar,
    )

    exact_results = await orchestrator.search(
        SearchRequest(
            query="fox query",
            corpus_ids=["corpus-1"],
            namespace="space-1",
        )
    )
    trigram_results = await orchestrator.search(
        SearchRequest(
            query="fox quary",
            corpus_ids=["corpus-1"],
            namespace="space-1",
        )
    )

    assert exact_results[0].id == "doc-exact"
    exact_sidecar = cast(dict[str, object], exact_results[0].metadata["search_sidecar"])
    trigram_sidecar = cast(dict[str, object], trigram_results[0].metadata["search_sidecar"])
    assert exact_results[0].score == 1.0
    assert trigram_results[0].score > 0.35
    assert exact_sidecar["strategy"] == "exact"
    assert trigram_sidecar["strategy"] == "trigram"


def test_check_health_delegates_to_store() -> None:
    asyncio.run(_run_check_health_delegates_to_store())


async def _run_check_health_delegates_to_store() -> None:
    store = RecordingVectorStore(health={"ok": True, "latency_ms": 12})
    orchestrator = SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    health = await orchestrator.check_health()

    assert health == {"ok": True, "latency_ms": 12}
