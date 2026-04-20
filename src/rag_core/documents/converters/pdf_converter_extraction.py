"""PDF page extraction helpers for the local converter path."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..pdf_text_quality import normalize_pdf_extracted_text

logger = logging.getLogger(__name__)

_MIN_CHARS_PER_PAGE = 50
_MIN_CHARS_IMAGE_PAGE = 200


def _is_encrypted_pdf_open_error(exc: Exception) -> bool:
    """Best-effort encrypted-PDF detection at the PyMuPDF boundary."""
    exc_name = type(exc).__name__.lower()
    if "password" in exc_name or "encrypted" in exc_name:
        return True
    exc_message = str(exc).lower()
    return "password" in exc_message or "encrypted" in exc_message


class PdfPageLike(Protocol):
    def get_text(self, mode: str) -> str: ...

    def get_images(self) -> Sequence[object]: ...


@dataclass
class PageExtraction:
    """Result of text extraction from a single PDF page."""

    page_num: int
    text: str = ""
    needs_ocr: bool = False
    char_count: int = 0
    has_garbled_text: bool = False


@dataclass
class PdfExtraction:
    """Result of extracting text from an entire PDF."""

    pages: list[PageExtraction] = field(default_factory=list)
    page_count: int = 0
    is_encrypted: bool = False

    @property
    def text_pages(self) -> list[PageExtraction]:
        """Pages with sufficient extracted text."""
        return [page for page in self.pages if not page.needs_ocr]

    @property
    def ocr_page_indices(self) -> list[int]:
        """0-based indices of pages needing OCR."""
        return [page.page_num for page in self.pages if page.needs_ocr]

    @property
    def full_text(self) -> str:
        """Combined text from all extracted pages (including partial)."""
        parts = [page.text for page in self.pages if page.text]
        return "\n\n".join(parts)

    @property
    def extraction_ratio(self) -> float:
        """Fraction of pages successfully extracted."""
        if not self.pages:
            return 0.0
        return len(self.text_pages) / len(self.pages)


def _extract_page(page: PdfPageLike, page_num: int) -> PageExtraction:
    """Extract text from a single PDF page with quality check."""
    try:
        raw_text = page.get_text("text") or ""
        text, has_garbled_text = normalize_pdf_extracted_text(raw_text)
        char_count = len(text)

        if has_garbled_text:
            return PageExtraction(
                page_num=page_num,
                text=text,
                needs_ocr=True,
                char_count=char_count,
                has_garbled_text=True,
            )

        if char_count < _MIN_CHARS_PER_PAGE:
            return PageExtraction(
                page_num=page_num,
                text=text,
                needs_ocr=True,
                char_count=char_count,
            )

        # Images present with minimal text suggests a scan.
        image_list = page.get_images()
        if image_list and char_count < _MIN_CHARS_IMAGE_PAGE:
            return PageExtraction(
                page_num=page_num,
                text=text,
                needs_ocr=True,
                char_count=char_count,
            )

        return PageExtraction(
            page_num=page_num,
            text=text,
            needs_ocr=False,
            char_count=char_count,
        )

    except Exception as exc:
        logger.warning("Text extraction failed for page %d: %s", page_num, exc)
        return PageExtraction(page_num=page_num, needs_ocr=True)


async def extract_pdf(file_bytes: bytes) -> PdfExtraction:
    """Extract text from PDF using PyMuPDF with per-page quality detection.

    Raises:
        ImportError: If PyMuPDF (fitz) is not installed.
    """
    import fitz

    def _extract() -> PdfExtraction:
        result = PdfExtraction()
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            if _is_encrypted_pdf_open_error(exc):
                result.is_encrypted = True
                return result
            raise

        try:
            if getattr(doc, "needs_pass", False):
                result.is_encrypted = True
                return result
            result.page_count = len(doc)
            for page_num in range(len(doc)):
                page = doc[page_num]
                extraction = _extract_page(page, page_num)
                result.pages.append(extraction)
        finally:
            doc.close()

        return result

    return await asyncio.to_thread(_extract)
