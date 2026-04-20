import asyncio

import pytest

from examples.minimal_app import run_demo as run_minimal_app_demo
from examples.corpus_lifecycle import (
    CorpusManifestEntry,
    delete_from_manifest,
    ingest_into_manifest,
    manifest_key,
    manifest_row,
    search_corpus,
)
from examples.pdf_ocr_path import describe_pdf_runtime, inspect_pdf_route, prepare_pdf_for_ingest
from rag_core import OcrRoutingSignal, ParsedDocument, PreparedDocument, PreparedChunk, RAGCore, RAGCoreConfig
from rag_core.documents import build_mistral_ocr_provider
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, RecordingVectorStore, make_search_result


def test_corpus_lifecycle_example_tracks_manifest_and_delete() -> None:
    asyncio.run(_run_corpus_lifecycle_example_tracks_manifest_and_delete())


async def _run_corpus_lifecycle_example_tracks_manifest_and_delete() -> None:
    store = RecordingVectorStore(
        search_results=[
            make_search_result(
                id="hit-1",
                text="billing answers stay searchable",
                score=0.92,
                document_id="doc-from-search",
                corpus_id="help-center",
            )
        ]
    )
    core = RAGCore(
        RAGCoreConfig(
            qdrant_location=":memory:",
            qdrant_collection="rag_core_examples_lifecycle",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    # Shape: {"acme:help-center:faq.txt": CorpusManifestEntry(document_id="doc_123", ...)}
    manifest: dict[str, CorpusManifestEntry] = {}

    try:
        entry = await ingest_into_manifest(
            core,
            manifest=manifest,
            file_bytes=b"billing answers stay searchable",
            filename="faq.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
            metadata={"source": "seed"},
        )
        key = manifest_key(
            namespace="acme",
            corpus_id="help-center",
            document_key=entry.document_key or "faq.txt",
        )

        assert manifest[key] == entry
        assert manifest_row(entry)["parser"] == "local:text"

        hits = await search_corpus(core, entry=entry, query="billing", limit=3)
        assert hits[0].document_id == "doc-from-search"
        assert store.search_calls[0].corpus_ids == ["help-center"]

        deleted = await delete_from_manifest(core, manifest=manifest, key=key)
        assert deleted.document_id == entry.document_id
        assert key not in manifest
        assert store.delete_calls[0].document_id == entry.document_id
    finally:
        await core.close()


def test_pdf_example_reports_route_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_pdf_example_reports_route_and_runtime(monkeypatch))


async def _run_pdf_example_reports_route_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parse_bytes(*, file_bytes: bytes, filename: str, mime_type: str, path: str | None = None) -> ParsedDocument:
        assert file_bytes == b"%PDF"
        assert filename == "scan.pdf"
        assert mime_type == "application/pdf"
        return ParsedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="",
            # Shape: {"parser": "local:pdf_inspector", "needs_ocr": True, "ocr_page_indices": [2, 0, 2]}
            metadata={"parser": "local:pdf_inspector", "needs_ocr": True, "ocr_page_indices": [2, 0, 2]},
            path=path,
        )

    async def fake_prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> PreparedDocument:
        assert file_bytes == b"%PDF"
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="# OCR text",
            # Shape: [PreparedChunk(chunk_index=0, text="# OCR text", embedding_text="# OCR text", word_count=3)]
            chunks=[PreparedChunk(chunk_index=0, text="# OCR text", embedding_text="# OCR text", word_count=3)],
            metadata={"parser": "local:pdf_inspector", "needs_ocr": False},
            path=path,
            ocr=OcrRoutingSignal(needed=False, page_indices=[], parser="local:pdf_inspector"),
        )

    core = RAGCore(
        RAGCoreConfig(
            qdrant_location=":memory:",
            qdrant_collection="rag_core_examples_pdf",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
        ocr_provider=build_mistral_ocr_provider(python_executable="/tmp/python"),
    )

    try:
        monkeypatch.setattr(core, "parse_bytes", fake_parse_bytes)
        monkeypatch.setattr(core, "prepare_bytes", fake_prepare_bytes)

        route = await inspect_pdf_route(core, file_bytes=b"%PDF", filename="scan.pdf", path="/tmp/scan.pdf")
        assert route == {
            "parser": "local:pdf_inspector",
            "needs_ocr": True,
            "ocr_page_indices": [0, 2],
        }

        prepared = await prepare_pdf_for_ingest(core, file_bytes=b"%PDF", filename="scan.pdf")
        assert prepared.markdown == "# OCR text"
        assert prepared.ocr.needed is False

        runtime = describe_pdf_runtime(core)
        assert runtime["ocr"] == {
            "provider": "mistral",
            "model": "mistral-ocr-latest",
            "supports_page_selection": True,
        }
        assert isinstance(runtime["pdf_inspector"], dict)
    finally:
        await core.close()


def test_minimal_app_demo_runs() -> None:
    asyncio.run(run_minimal_app_demo())
