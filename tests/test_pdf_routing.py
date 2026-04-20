import asyncio

import rag_core.core as core_module
from rag_core.core_models import ParsedDocument
from rag_core.core_prepare import apply_ocr
from rag_core.core_models import OcrRoutingSignal, PreparedChunk, PreparedDocument
import rag_core.documents.converters.pdf_converter as pdf_converter_module
from rag_core import RAGCore, RAGCoreConfig
from rag_core.documents.converters.pdf_converter import PdfConverter
from rag_core.documents.ocr import OcrRequest, OcrResult
from rag_core.documents.pdf_inspector import (
    PdfInspectorDetectionResult,
    PdfInspectorExtractionResult,
    pdf_inspector_enabled,
)

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, RecordingVectorStore


def test_pdf_inspector_enabled_defaults_to_true(monkeypatch) -> None:
    monkeypatch.delenv("PDF_INSPECTOR_MODE", raising=False)

    assert pdf_inspector_enabled() is True


def test_pdf_converter_prefers_inspector_for_text_pdfs(monkeypatch) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=("canonical inspector markdown " * 8).strip(),
    )

    async def fail_pymupdf(self, file_bytes: bytes, filename: str, mime_type: str):
        raise AssertionError("PyMuPDF fallback should not run when inspector returns canonical text")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", lambda file_bytes: detection)
    monkeypatch.setattr(pdf_converter_module, "extract_pdf_with_inspector", lambda file_bytes: extraction)
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fail_pymupdf)

    result = asyncio.run(PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf"))

    assert result.content == extraction.markdown
    assert result.metadata["parser"] == "local:pdf_inspector"
    assert result.metadata["inspector_route"] == "text"
    assert result.metadata["needs_ocr"] is False
    assert result.metadata["ocr_page_count"] == 0
    assert result.needs_ocr is False


def test_pdf_converter_emits_explicit_ocr_routing_metadata_for_mixed_pdfs(monkeypatch) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=4,
        pages_needing_ocr=[],
        confidence=0.81,
        has_encoding_issues=False,
        processing_time_ms=7,
        is_complex=True,
        pages_with_tables=[2],
        pages_with_columns=[1, 2],
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=4,
        pages_needing_ocr=[2, 0, 2, -1, 9],
        has_encoding_issues=False,
        processing_time_ms=13,
        markdown=("mixed inspector markdown " * 12).strip(),
        is_complex=True,
        pages_with_tables=[2],
        pages_with_columns=[1],
    )

    async def fail_pymupdf(self, file_bytes: bytes, filename: str, mime_type: str):
        raise AssertionError("PyMuPDF fallback should not run when inspector supports mixed routing")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", lambda file_bytes: detection)
    monkeypatch.setattr(pdf_converter_module, "extract_pdf_with_inspector", lambda file_bytes: extraction)
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fail_pymupdf)

    result = asyncio.run(PdfConverter().convert(b"%PDF-1.7", "mixed.pdf", "application/pdf"))

    assert result.metadata["parser"] == "local:pdf_inspector"
    assert result.metadata["inspector_route"] == "mixed"
    assert result.metadata["needs_ocr"] is True
    assert sorted(result.metadata["ocr_page_indices"]) == [0, 2]
    assert result.metadata["ocr_page_count"] == 2
    assert result.metadata["complex_ocr_page_indices"] == [2]
    assert result.metadata["extraction_ratio"] == 0.5
    assert result.needs_ocr is True
    assert sorted(result.ocr_page_indices or []) == [0, 2]


def test_prepare_bytes_surfaces_ocr_routing_signal(monkeypatch) -> None:
    asyncio.run(_run_prepare_bytes_surfaces_ocr_routing_signal(monkeypatch))


async def _run_prepare_bytes_surfaces_ocr_routing_signal(monkeypatch) -> None:
    async def fake_prepare_document_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None,
        contextualize: bool,
        ocr_provider,
    ):
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="parsed markdown",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="parsed markdown",
                    embedding_text="parsed markdown",
                    word_count=2,
                )
            ],
            metadata={
                "needs_ocr": True,
                "ocr_page_indices": [2, 0, 2],
                "parser": "local:pdf_inspector",
                "confidence": "0.62",
            },
            ocr=OcrRoutingSignal(
                needed=True,
                page_indices=[0, 2],
                confidence=0.62,
                parser="local:pdf_inspector",
            ),
        )

    monkeypatch.setattr(core_module, "prepare_document_bytes", fake_prepare_document_bytes)

    core = RAGCore(
        RAGCoreConfig(
            qdrant_location=":memory:",
            embedding_dimensions=4,
            contextualize=False,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    try:
        prepared = await core.prepare_bytes(
            file_bytes=b"%PDF-1.7",
            filename="report.pdf",
            mime_type="application/pdf",
        )
    finally:
        await core.close()

    assert [chunk.embedding_text for chunk in prepared.chunks] == ["parsed markdown"]
    assert prepared.ocr.needed is True
    assert prepared.ocr.page_indices == [0, 2]
    assert prepared.ocr.confidence == 0.62
    assert prepared.ocr.parser == "local:pdf_inspector"


def test_apply_ocr_replaces_markdown_for_full_document_helpers() -> None:
    asyncio.run(_run_apply_ocr_replaces_markdown_for_full_document_helpers())


async def _run_apply_ocr_replaces_markdown_for_full_document_helpers() -> None:
    class FakeOcrProvider:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown="# OCR Full Document",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
                metadata={
                    "ocr_page_indices_ignored": True,
                    "ocr_processed_entire_document": True,
                },
            )

    result = await apply_ocr(
        parsed=ParsedDocument(
            filename="scan.pdf",
            mime_type="application/pdf",
            markdown="# Local Extracted Text",
            metadata={"needs_ocr": True, "ocr_page_indices": [0], "page_count": 4},
        ),
        file_bytes=b"%PDF-1.7",
        provider=FakeOcrProvider(),
    )

    assert result.markdown == "# OCR Full Document"
    assert result.metadata["ocr_merge_mode"] == "replace"
    assert result.metadata["ocr_pages_used"] == [0, 1, 2, 3]
    assert result.metadata["ocr_pages_used_count"] == 4
    assert result.metadata["needs_ocr"] is False


def test_apply_ocr_appends_markdown_for_partial_page_ocr() -> None:
    asyncio.run(_run_apply_ocr_appends_markdown_for_partial_page_ocr())


async def _run_apply_ocr_appends_markdown_for_partial_page_ocr() -> None:
    class FakeOcrProvider:
        provider_name = "mistral"
        model_name = "mistral-ocr-latest"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [2]
            return OcrResult(
                markdown="## OCR Page 3",
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[2],
            )

    result = await apply_ocr(
        parsed=ParsedDocument(
            filename="scan.pdf",
            mime_type="application/pdf",
            markdown="# Local Extracted Text",
            metadata={"needs_ocr": True, "ocr_page_indices": [2]},
        ),
        file_bytes=b"%PDF-1.7",
        provider=FakeOcrProvider(),
    )

    assert result.markdown == "# Local Extracted Text\n\n## OCR Page 3"
    assert result.metadata["ocr_merge_mode"] == "append"
    assert result.metadata["ocr_page_indices"] == [2]
