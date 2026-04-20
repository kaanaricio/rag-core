import asyncio

from rag_core.search.indexer import IndexRequest, QdrantIndexer
from rag_core.search.indexer_points import make_point_id

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    FakeSparseEmbedderNoMulti,
    RecordingVectorStore,
)


def test_index_document_uses_payload_chunks_and_contextual_dense_text() -> None:
    asyncio.run(_run_index_document_uses_payload_chunks_and_contextual_dense_text())


async def _run_index_document_uses_payload_chunks_and_contextual_dense_text() -> None:
    embedding = FakeEmbeddingProvider(vocabulary=("original", "context", "fox"))
    store = RecordingVectorStore()
    indexer = QdrantIndexer(
        embedding_provider=embedding,
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    result = await indexer.index_document(
        IndexRequest(
            document_id="doc-1",
            corpus_id="corpus-1",
            namespace=" team-space ",
            text="unused",
            filename="report.txt",
            mime_type="text/plain",
            source_type="file",
            path="/docs/report.txt",
            extra_fields={"team": "search"},
            pre_chunked_texts=["original fox"],
            embedding_chunk_texts=["context fox"],
        )
    )

    point = store.upsert_calls[0][0]

    assert result.chunk_count == 1
    assert store.operations == ["upsert"]
    assert store.delete_calls == []
    assert point.payload["namespace"] == "team-space"
    assert "original fox" in str(point.payload["text"])
    assert "context fox" not in str(point.payload["text"])
    assert "**Source Type**: file" in str(point.payload["text"])
    assert point.dense_vector == [0.0, 1.0, 1.0]


def test_index_document_falls_back_when_embedding_chunk_lengths_do_not_match() -> None:
    asyncio.run(_run_index_document_falls_back_when_embedding_chunk_lengths_do_not_match())


async def _run_index_document_falls_back_when_embedding_chunk_lengths_do_not_match() -> None:
    embedding = FakeEmbeddingProvider(vocabulary=("original", "context", "fox"))
    store = RecordingVectorStore()
    indexer = QdrantIndexer(
        embedding_provider=embedding,
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    await indexer.index_document(
        IndexRequest(
            document_id="doc-1",
            corpus_id="corpus-1",
            namespace="team-space",
            text="unused",
            filename="report.txt",
            mime_type="text/plain",
            source_type="file",
            pre_chunked_texts=["original fox"],
            embedding_chunk_texts=["context fox", "extra"],
        )
    )

    point = store.upsert_calls[0][0]
    assert point.dense_vector == [1.0, 0.0, 1.0]


def test_index_document_falls_back_to_bm25_only_when_multi_channel_is_unavailable() -> None:
    asyncio.run(_run_index_document_falls_back_to_bm25_only_when_multi_channel_is_unavailable())


async def _run_index_document_falls_back_to_bm25_only_when_multi_channel_is_unavailable() -> None:
    store = RecordingVectorStore()
    indexer = QdrantIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedderNoMulti(),
        vector_store=store,
    )

    await indexer.index_document(
        IndexRequest(
            document_id="doc-1",
            corpus_id="corpus-1",
            namespace="team-space",
            text="unused",
            filename="report.txt",
            mime_type="text/plain",
            source_type="file",
            pre_chunked_texts=["fox query"],
        )
    )

    point = store.upsert_calls[0][0]
    assert set(point.sparse_vectors) == {"bm25"}
    assert point.sparse_vector == point.sparse_vectors["bm25"]


def test_index_document_builds_section_payload_from_mappings() -> None:
    asyncio.run(_run_index_document_builds_section_payload_from_mappings())


async def _run_index_document_builds_section_payload_from_mappings() -> None:
    store = RecordingVectorStore()
    indexer = QdrantIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
        vector_store=store,
    )

    await indexer.index_document(
        IndexRequest(
            document_id="doc-1",
            corpus_id="corpus-1",
            namespace="team-space",
            text="unused",
            filename="slides.pdf",
            mime_type="application/pdf",
            source_type="file",
            pre_chunked_texts=["page one", "page two"],
            section_mappings=[
                {
                    "chunk_index": 1,
                    "section_id": "sec-2",
                    "section_path": "Intro > Details",
                    "page_number": 3,
                    "result_type": "image",
                    "thumbnail_url": "thumb.png",
                }
            ],
        )
    )

    point = store.upsert_calls[0][1]
    assert point.payload["section_id"] == "sec-2"
    assert point.payload["section_path"] == "Intro > Details"
    assert point.payload["section_title"] == "Details"
    assert point.payload["page_number"] == 3
    assert point.payload["result_type"] == "image"
    assert point.payload["thumbnail_url"] == "thumb.png"


def test_index_document_prefers_metadata_title_for_display_title() -> None:
    asyncio.run(_run_index_document_prefers_metadata_title_for_display_title())


async def _run_index_document_prefers_metadata_title_for_display_title() -> None:
    store = RecordingVectorStore()
    indexer = QdrantIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
        vector_store=store,
    )

    await indexer.index_document(
        IndexRequest(
            document_id="doc-1",
            corpus_id="corpus-1",
            namespace="team-space",
            text="unused",
            filename="report.txt",
            mime_type="text/plain",
            source_type="file",
            extra_fields={"title": "Quarterly Report"},
            pre_chunked_texts=["page one"],
        )
    )

    point = store.upsert_calls[0][0]
    assert point.payload["title"] == "Quarterly Report"


def test_index_document_requires_non_blank_namespace() -> None:
    asyncio.run(_run_index_document_requires_non_blank_namespace())


async def _run_index_document_requires_non_blank_namespace() -> None:
    indexer = QdrantIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    try:
        await indexer.index_document(
            IndexRequest(
                document_id="doc-1",
                corpus_id="corpus-1",
                namespace="   ",
                text="unused",
                filename="report.txt",
                mime_type="text/plain",
                source_type="file",
                pre_chunked_texts=["fox query"],
            )
        )
    except ValueError as exc:
        assert str(exc) == "namespace is required for indexing"
    else:
        raise AssertionError("Expected namespace validation to fail")


def test_delete_document_requires_non_blank_namespace() -> None:
    asyncio.run(_run_delete_document_requires_non_blank_namespace())


async def _run_delete_document_requires_non_blank_namespace() -> None:
    indexer = QdrantIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    try:
        await indexer.delete_document(document_id="doc-1", namespace=" ", corpus_id="corpus-1")
    except ValueError as exc:
        assert str(exc) == "namespace is required for delete_document"
    else:
        raise AssertionError("Expected delete namespace validation to fail")


def test_index_document_deletes_only_stale_tail_chunks_after_upsert() -> None:
    asyncio.run(_run_index_document_deletes_only_stale_tail_chunks_after_upsert())


async def _run_index_document_deletes_only_stale_tail_chunks_after_upsert() -> None:
    store = RecordingVectorStore()
    indexer = QdrantIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    result = await indexer.index_document(
        IndexRequest(
            document_id="doc-1",
            corpus_id="corpus-1",
            namespace="team-space",
            text="unused",
            filename="report.txt",
            mime_type="text/plain",
            source_type="file",
            existing_chunk_count=3,
            pre_chunked_texts=["page one"],
        )
    )

    assert result.chunk_count == 1
    assert store.operations == ["upsert", "delete_point_ids"]
    assert store.delete_calls == []
    assert store.delete_point_ids_calls == [
        [
            make_point_id(
                namespace="team-space",
                corpus_id="corpus-1",
                document_id="doc-1",
                chunk_index=1,
            ),
            make_point_id(
                namespace="team-space",
                corpus_id="corpus-1",
                document_id="doc-1",
                chunk_index=2,
            ),
        ]
    ]


def test_make_point_id_scopes_namespace_and_corpus() -> None:
    first = make_point_id(
        namespace="team-a",
        corpus_id="corpus-1",
        document_id="doc-1",
        chunk_index=0,
    )
    second = make_point_id(
        namespace="team-b",
        corpus_id="corpus-1",
        document_id="doc-1",
        chunk_index=0,
    )
    third = make_point_id(
        namespace="team-a",
        corpus_id="corpus-2",
        document_id="doc-1",
        chunk_index=0,
    )

    assert first != second
    assert first != third
