"""PDF converter with hybrid text extraction + OCR fallback.

Improvements over AirWeave:
- Per-page quality scoring (not just char count < 50)
- Partial OCR: only sends image-only pages to OCR, keeps text pages as-is
  (AirWeave sends entire PDF to OCR if any page lacks text)
- Password-protected PDF handling
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from ..pdf_inspector import (
    detect_pdf_with_inspector,
    extract_pdf_with_inspector,
    pdf_inspector_enabled,
)
from .base import (
    ConversionResult,
    HybridConverter,
    QualityScore,
    QualityVerdict,
    score_text_quality,
    text_to_markdown,
)
from .pdf_converter_extraction import PageExtraction, PdfExtraction, extract_pdf
from .pdf_converter_inspector import (
    _apply_inspector_analysis_metadata,
    _get_inspector_field,
    _get_inspector_markdown,
    _get_inspector_metadata,
    _get_inspector_page_count,
    _get_inspector_route,
    _inspector_is_ocr_only_route,
    _inspector_is_text_based,
    _inspector_supports_page_level_routing,
    _normalize_inspector_ocr_page_indices,
)

logger = logging.getLogger(__name__)


class PdfConverter(HybridConverter):
    """Converts PDFs to markdown with per-page OCR tracking.

    Key advantage over AirWeave: supports PARTIAL OCR.
    Instead of sending the entire PDF to OCR when any page lacks text,
    we track exactly which pages need OCR and which have good text.
    The ingest pipeline can then OCR only the pages that need it.
    """

    format_name = "pdf"

    async def _try_extract(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        inspector_result = await self._try_extract_with_inspector(file_bytes, filename)
        if inspector_result is not None:
            return inspector_result

        return await self._try_extract_with_pymupdf(file_bytes, filename, mime_type)

    async def _try_extract_with_inspector(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Optional[ConversionResult]:
        try:
            if not pdf_inspector_enabled():
                return None
        except Exception as exc:
            logger.warning("PDF Inspector availability check failed for %s: %s", filename, exc)
            return None

        try:
            detection = await asyncio.to_thread(detect_pdf_with_inspector, file_bytes)
        except Exception as exc:
            logger.warning("PDF Inspector detection failed for %s: %s", filename, exc)
            return None
        route = _get_inspector_route(detection)
        page_count = _get_inspector_page_count(detection) or 1
        if _inspector_is_ocr_only_route(detection):
            normalized_ocr_indices = _normalize_inspector_ocr_page_indices(
                _get_inspector_field(detection, "pages_needing_ocr"),
                page_count=page_count,
                default_all_pages=True,
            )
            metadata: Dict[str, Any] = {}
            metadata.update(_get_inspector_metadata(detection))
            metadata.update(
                {
                    "parser": "local:pdf_inspector",
                    "page_count": page_count,
                    "needs_ocr": True,
                    "extraction_ratio": 0.0,
                    "ocr_page_count": len(normalized_ocr_indices),
                }
            )
            if normalized_ocr_indices:
                metadata["ocr_page_indices"] = normalized_ocr_indices
            _apply_inspector_analysis_metadata(
                metadata,
                detection=detection,
                extraction=None,
                ocr_page_indices=normalized_ocr_indices,
            )

            return ConversionResult(
                content="",
                metadata=metadata,
                quality=QualityScore(
                    page_count=page_count,
                    verdict=QualityVerdict.EMPTY,
                    details="pdf inspector classified document as OCR-only",
                ),
                needs_ocr=True,
                ocr_page_indices=normalized_ocr_indices or None,
            )

        if not _inspector_is_text_based(detection) and not _inspector_supports_page_level_routing(
            detection
        ):
            logger.info(
                "PDF Inspector classified %s as %s, falling back to PyMuPDF path",
                filename,
                route or "unknown",
            )
            return None

        try:
            extraction = await asyncio.to_thread(extract_pdf_with_inspector, file_bytes)
        except Exception as exc:
            logger.warning("PDF Inspector extraction failed for %s: %s", filename, exc)
            return None

        markdown = _get_inspector_markdown(extraction)
        if _inspector_is_text_based(detection):
            if not markdown:
                logger.info(
                    "PDF Inspector returned no markdown for %s, falling back to PyMuPDF path",
                    filename,
                )
                return None

            page_count = _get_inspector_page_count(extraction, detection) or page_count
            quality = score_text_quality(
                markdown,
                page_count=page_count,
                min_chars=1,
                min_chars_per_page=1.0,
            )
            quality.verdict = QualityVerdict.GOOD
            quality.details = "pdf inspector canonical extraction"

            metadata = {}
            metadata.update(_get_inspector_metadata(detection))
            metadata.update(_get_inspector_metadata(extraction))
            metadata.update(
                {
                    "parser": "local:pdf_inspector",
                    "page_count": page_count,
                    "needs_ocr": False,
                    "extraction_ratio": 1.0,
                    "ocr_page_count": 0,
                }
            )
            _apply_inspector_analysis_metadata(
                metadata,
                detection=detection,
                extraction=extraction,
                ocr_page_indices=[],
            )

            return ConversionResult(
                content=markdown,
                metadata=metadata,
                quality=quality,
            )

        if not _inspector_supports_page_level_routing(detection):
            logger.info(
                "PDF Inspector classified %s as %s, falling back to PyMuPDF path",
                filename,
                route or "unknown",
            )
            return None
        if not markdown:
            logger.info(
                "PDF Inspector returned no markdown for mixed PDF %s, falling back to PyMuPDF path",
                filename,
            )
            return None

        page_count = _get_inspector_page_count(extraction, detection) or page_count
        normalized_ocr_indices = _normalize_inspector_ocr_page_indices(
            _get_inspector_field(extraction, "pages_needing_ocr"),
            page_count=page_count,
        )
        quality = score_text_quality(
            markdown,
            page_count=page_count,
            min_chars=1,
            min_chars_per_page=1.0,
        )
        if markdown:
            quality.details = "pdf inspector mixed extraction"
            if quality.verdict == QualityVerdict.EMPTY:
                quality.verdict = QualityVerdict.POOR

        metadata = {}
        metadata.update(_get_inspector_metadata(detection))
        metadata.update(_get_inspector_metadata(extraction))
        metadata.update(
            {
                "parser": "local:pdf_inspector",
                "page_count": page_count,
                "needs_ocr": bool(normalized_ocr_indices),
                "extraction_ratio": max(
                    0.0,
                    (page_count - len(normalized_ocr_indices)) / page_count if page_count else 0.0,
                ),
                "ocr_page_count": len(normalized_ocr_indices),
            }
        )
        if normalized_ocr_indices:
            metadata["ocr_page_indices"] = normalized_ocr_indices
        _apply_inspector_analysis_metadata(
            metadata,
            detection=detection,
            extraction=extraction,
            ocr_page_indices=normalized_ocr_indices,
        )

        return ConversionResult(
            content=markdown,
            metadata=metadata,
            quality=quality,
            needs_ocr=bool(normalized_ocr_indices),
            ocr_page_indices=normalized_ocr_indices or None,
        )

    async def _try_extract_with_pymupdf(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Extract text from PDF with per-page quality scoring."""
        extraction = await extract_pdf(file_bytes)

        metadata: Dict[str, Any] = {
            "parser": "local:pymupdf",
            "page_count": extraction.page_count,
        }

        if extraction.is_encrypted:
            logger.info("PDF %s is password-protected, needs OCR", filename)
            return ConversionResult(
                needs_ocr=True,
                metadata={**metadata, "is_encrypted": True},
                quality=QualityScore(verdict=QualityVerdict.EMPTY, details="encrypted PDF"),
            )

        if not extraction.pages:
            return ConversionResult(
                metadata=metadata,
                quality=QualityScore(verdict=QualityVerdict.EMPTY, details="no pages"),
            )

        sections: list[str] = []
        for page in extraction.pages:
            if page.text:
                markdown = text_to_markdown(page.text)
                sections.append("## Page %d\n\n%s" % (page.page_num + 1, markdown))

        content = "\n\n".join(sections) if sections else ""

        quality = score_text_quality(
            content,
            page_count=extraction.page_count,
            min_chars=50,
            min_chars_per_page=20.0,
        )

        ocr_indices = extraction.ocr_page_indices
        garbled_page_indices = [
            page.page_num for page in extraction.pages if getattr(page, "has_garbled_text", False)
        ]
        metadata["needs_ocr"] = bool(ocr_indices)
        metadata["extraction_ratio"] = extraction.extraction_ratio
        metadata["ocr_page_count"] = len(ocr_indices)
        if garbled_page_indices:
            metadata["garbled_text_page_indices"] = garbled_page_indices

        if ocr_indices:
            logger.info(
                "PDF %s: %d/%d pages need OCR (partial OCR possible)",
                filename,
                len(ocr_indices),
                extraction.page_count,
            )

        return ConversionResult(
            content=content,
            metadata=metadata,
            quality=quality,
            needs_ocr=bool(ocr_indices) and extraction.extraction_ratio < 0.5,
            ocr_page_indices=ocr_indices if ocr_indices else None,
        )
