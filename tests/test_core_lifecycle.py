import asyncio
from typing import cast

from rag_core.core_models import IngestedDocument, OcrRoutingSignal, PreparedChunk, PreparedDocument
from rag_core import RAGCore, RAGCoreConfig
from rag_core.core_lifecycle import compute_content_sha256
from rag_core.search.lexical_sidecar import LexicalSidecarRecord, PortableLexicalSidecar
from rag_core.search.types import SearchSidecarQuery, StoredDocumentRecord

from tests.support import (
    FakeEmbeddingProvider,
    FakeSearchSidecar,
    FakeSparseEmbedder,
    RecordingVectorStore,
)


def test_ingest_uses_stable_document_key_and_builds_manifest() -> None:
    asyncio.run(_run_ingest_uses_stable_document_key_and_builds_manifest())


async def _run_ingest_uses_stable_document_key_and_builds_manifest() -> None:
    store = RecordingVectorStore()
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        doc = await core.ingest_bytes(
            file_bytes=b"alpha fox query",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            path="/docs/guide.txt",
        )
        manifest = core.build_corpus_manifest(
            namespace="team-space",
            corpus_id="corpus-1",
            documents=[doc],
        )
    finally:
        await core.close()

    assert doc.document_key == "/docs/guide.txt"
    assert doc.document_id.startswith("doc_")
    assert doc.content_sha256 is not None
    assert doc.ingest_state == "created"
    assert doc.collection_name == "rag_core_chunks__fake_embedding_4d"
    assert store.get_document_record_calls == [("team-space", "corpus-1", doc.document_id, None)]
    assert manifest.document_count == 1
    assert manifest.chunk_count == doc.chunk_count
    assert manifest.source_document_ids == (doc.document_id,)
    assert manifest.embedding_model == "fake-embedding"
    assert manifest.entries[0].document_key == doc.document_key
    assert manifest.entries[0].content_sha256 == doc.content_sha256


def test_ingest_skips_reindex_when_content_is_unchanged() -> None:
    asyncio.run(_run_ingest_skips_reindex_when_content_is_unchanged())


async def _run_ingest_skips_reindex_when_content_is_unchanged() -> None:
    existing = StoredDocumentRecord(
        document_id="doc_unchanged",
        namespace="team-space",
        corpus_id="corpus-1",
        document_key="/docs/guide.txt",
        content_sha256=compute_content_sha256(b"same bytes"),
        chunk_count=3,
    )
    store = RecordingVectorStore(
        document_records={("team-space", "corpus-1", "doc_unchanged"): existing},
    )
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        doc = await core.ingest_bytes(
            file_bytes=b"same bytes",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            document_id="doc_unchanged",
            path="/docs/guide.txt",
        )
    finally:
        await core.close()

    assert doc.ingest_state == "unchanged"
    assert doc.chunk_count == 3
    assert doc.metadata["parser"] == "local:text"
    assert doc.ocr.needed is False
    assert store.upsert_calls == []


def test_ingest_returns_caller_metadata_on_indexed_and_unchanged_paths() -> None:
    asyncio.run(_run_ingest_returns_caller_metadata_on_indexed_and_unchanged_paths())


async def _run_ingest_returns_caller_metadata_on_indexed_and_unchanged_paths() -> None:
    unchanged_record = StoredDocumentRecord(
        document_id="doc-existing",
        namespace="team-space",
        corpus_id="corpus-1",
        document_key="/docs/guide.txt",
        content_sha256=compute_content_sha256(b"same bytes"),
        chunk_count=1,
    )
    store = RecordingVectorStore(
        document_records={("team-space", "corpus-1", "doc-existing"): unchanged_record},
    )
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        created = await core.ingest_bytes(
            file_bytes=b"new bytes",
            filename="created.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            path="/docs/created.txt",
            metadata={"title": "Created Title"},
        )
        unchanged = await core.ingest_bytes(
            file_bytes=b"same bytes",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            document_id="doc-existing",
            path="/docs/guide.txt",
            metadata={"title": "Guide Title"},
        )
    finally:
        await core.close()

    assert created.metadata["title"] == "Created Title"
    assert unchanged.metadata["title"] == "Guide Title"


def test_unchanged_ingest_does_not_mutate_sidecar_records() -> None:
    asyncio.run(_run_unchanged_ingest_does_not_mutate_sidecar_records())


async def _run_unchanged_ingest_does_not_mutate_sidecar_records() -> None:
    existing = StoredDocumentRecord(
        document_id="doc_unchanged",
        namespace="team-space",
        corpus_id="corpus-1",
        document_key="/docs/guide.txt",
        content_sha256=compute_content_sha256(b"same bytes"),
        chunk_count=3,
    )
    sidecar = FakeSearchSidecar()
    store = RecordingVectorStore(
        document_records={("team-space", "corpus-1", "doc_unchanged"): existing},
    )
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        search_sidecar=sidecar,
    )

    try:
        await core.ingest_bytes(
            file_bytes=b"same bytes",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            document_id="doc_unchanged",
            path="/docs/guide.txt",
        )
    finally:
        await core.close()

    assert sidecar.upserted == []
    assert sidecar.deleted == []


def test_ingest_marks_existing_document_as_replaced_when_hash_changes() -> None:
    asyncio.run(_run_ingest_marks_existing_document_as_replaced_when_hash_changes())


async def _run_ingest_marks_existing_document_as_replaced_when_hash_changes() -> None:
    store = RecordingVectorStore(
        document_records={
            ("team-space", "corpus-1", "doc_existing"): StoredDocumentRecord(
                document_id="doc_existing",
                namespace="team-space",
                corpus_id="corpus-1",
                document_key="/docs/guide.txt",
                content_sha256="old-hash",
                chunk_count=1,
            )
        }
    )
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        doc = await core.ingest_bytes(
            file_bytes=b"new bytes",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            document_id="doc_existing",
            path="/docs/guide.txt",
        )
    finally:
        await core.close()

    assert doc.ingest_state == "replaced"
    assert doc.replaced_existing is True
    assert store.operations[:2] == ["get_document_record", "upsert"]


def test_ingest_and_delete_keep_sidecar_in_sync() -> None:
    asyncio.run(_run_ingest_and_delete_keep_sidecar_in_sync())


async def _run_ingest_and_delete_keep_sidecar_in_sync() -> None:
    sidecar = FakeSearchSidecar()
    store = RecordingVectorStore()
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        search_sidecar=sidecar,
    )

    try:
        doc = await core.ingest_bytes(
            file_bytes=b"alpha fox query",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            path="/docs/guide.txt",
        )
        await core.delete_document(
            document_id=doc.document_id,
            namespace="team-space",
            corpus_id="corpus-1",
        )
    finally:
        await core.close()

    assert len(sidecar.upserted) == doc.chunk_count
    first_record = cast(LexicalSidecarRecord, sidecar.upserted[0])
    assert "**Source Type**: file" in first_record.result.text
    assert first_record.result.document_key == doc.document_key
    assert first_record.result.content_sha256 == doc.content_sha256
    assert first_record.result.chunk_word_count is not None
    assert first_record.result.chunker_strategy == "prechunked"
    assert first_record.result.result_type == "text"
    assert sidecar.deleted == [
        ("team-space", doc.document_id),
        ("team-space", doc.document_id),
    ]


def test_delete_document_requires_corpus_id() -> None:
    asyncio.run(_run_delete_document_requires_corpus_id())


async def _run_delete_document_requires_corpus_id() -> None:
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    try:
        try:
            await core.delete_document(document_id="doc-1", namespace="team-space", corpus_id="")
        except ValueError as exc:
            assert str(exc) == "corpus_id is required for delete_document"
        else:
            raise AssertionError("Expected corpus_id validation to fail")
    finally:
        await core.close()


def test_explicit_document_ids_are_scoped_by_corpus_for_existing_checks() -> None:
    asyncio.run(_run_explicit_document_ids_are_scoped_by_corpus_for_existing_checks())


async def _run_explicit_document_ids_are_scoped_by_corpus_for_existing_checks() -> None:
    store = RecordingVectorStore(
        document_records={
            ("team-space", "corpus-2", "doc-shared"): StoredDocumentRecord(
                document_id="doc-shared",
                namespace="team-space",
                corpus_id="corpus-2",
                document_key="/docs/guide.txt",
                content_sha256=compute_content_sha256(b"same bytes"),
                chunk_count=5,
            )
        }
    )
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        doc = await core.ingest_bytes(
            file_bytes=b"same bytes",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            document_id="doc-shared",
            path="/docs/guide.txt",
        )
    finally:
        await core.close()

    assert doc.ingest_state == "created"
    assert store.get_document_record_calls == [("team-space", "corpus-1", "doc-shared", None)]


def test_manifest_file_is_preview_only() -> None:
    asyncio.run(_run_manifest_file_is_preview_only())


async def _run_manifest_file_is_preview_only() -> None:
    from pathlib import Path
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    try:
        handle.write(b"preview only")
        handle.flush()
        file_path = Path(handle.name)
    finally:
        handle.close()

    store = RecordingVectorStore()
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    try:
        entry = await core.manifest_file(
            file_path,
            namespace="team-space",
            corpus_id="corpus-1",
        )
    finally:
        await core.close()
        file_path.unlink(missing_ok=True)

    assert entry.document_key == str(file_path)
    assert store.upsert_calls == []
    assert store.delete_calls == []


def test_reingest_shrinks_sidecar_results_without_stale_chunks() -> None:
    asyncio.run(_run_reingest_shrinks_sidecar_results_without_stale_chunks())


async def _run_reingest_shrinks_sidecar_results_without_stale_chunks() -> None:
    sidecar = PortableLexicalSidecar([])
    store = RecordingVectorStore()
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        search_sidecar=sidecar,
    )

    async def fake_prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> PreparedDocument:
        if file_bytes == b"first":
            return PreparedDocument(
                filename=filename,
                mime_type=mime_type,
                markdown="alpha\n\nbeta",
                chunks=[
                    PreparedChunk(chunk_index=0, text="alpha", embedding_text="alpha", word_count=1),
                    PreparedChunk(chunk_index=1, text="beta", embedding_text="beta", word_count=1),
                ],
                metadata={"parser": "local:text"},
                path=path,
                ocr=OcrRoutingSignal(),
            )
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="alpha",
            chunks=[
                PreparedChunk(chunk_index=0, text="alpha", embedding_text="alpha", word_count=1),
            ],
            metadata={"parser": "local:text"},
            path=path,
            ocr=OcrRoutingSignal(),
        )

    core.prepare_bytes = fake_prepare_bytes  # type: ignore[method-assign]

    try:
        first = await core.ingest_bytes(
            file_bytes=b"first",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            path="/docs/guide.txt",
        )
        second = await core.ingest_bytes(
            file_bytes=b"second",
            filename="guide.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
            document_id=first.document_id,
            path="/docs/guide.txt",
        )
        removed = await sidecar.search(
            SearchSidecarQuery(
                query="beta",
                namespace="team-space",
                corpus_ids=["corpus-1"],
            )
        )
    finally:
        await core.close()

    assert second.chunk_count == 1
    assert removed == []


def test_corpus_manifest_counts_ocr_usage_not_pending_state() -> None:
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    document = IngestedDocument(
        document_id="doc-1",
        namespace="team-space",
        corpus_id="corpus-1",
        chunk_count=1,
        filename="scan.pdf",
        mime_type="application/pdf",
        document_key="/docs/scan.pdf",
        content_sha256="hash-1",
        metadata={
            "parser": "local:pdf_inspector",
            "needs_ocr": False,
            "ocr_provider_used": True,
            "ocr_pages_used": [0, 2],
        },
        ocr=OcrRoutingSignal(needed=False, page_indices=[]),
    )
    used_ocr = core.build_manifest_entry(document=document)
    manifest = core.build_corpus_manifest(
        namespace="team-space",
        corpus_id="corpus-1",
        documents=[document],
    )

    assert used_ocr.needs_ocr is False
    assert manifest.ocr_document_count == 1
    assert manifest.ocr_page_count == 2


def test_manifest_uses_explicit_ocr_page_count_when_present() -> None:
    core = RAGCore(
        RAGCoreConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    document = IngestedDocument(
        document_id="doc-1",
        namespace="team-space",
        corpus_id="corpus-1",
        chunk_count=1,
        filename="scan.pdf",
        mime_type="application/pdf",
        document_key="/docs/scan.pdf",
        content_sha256="hash-1",
        metadata={
            "parser": "local:pdf_inspector",
            "needs_ocr": False,
            "ocr_provider_used": True,
            "ocr_pages_used_count": 4,
            "ocr_pages_used": [0],
        },
        ocr=OcrRoutingSignal(needed=False, page_indices=[]),
    )
    try:
        manifest = core.build_corpus_manifest(
            namespace="team-space",
            corpus_id="corpus-1",
            documents=[document],
        )
    finally:
        asyncio.run(core.close())

    assert manifest.ocr_document_count == 1
    assert manifest.ocr_page_count == 4
