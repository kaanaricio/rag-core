"""Base converter interfaces and quality scoring for document conversion.

Provides:
- BaseConverter: protocol for all converters
- HybridConverter: extract-first / OCR-fallback orchestration
- QualityScore: multi-signal quality assessment (beats AirWeave's 50-char threshold)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)




class QualityVerdict(str, Enum):
    """Whether extracted text is good enough or OCR is needed."""

    GOOD = "good"
    POOR = "poor"
    EMPTY = "empty"


@dataclass
class QualityScore:
    """Multi-signal quality assessment of extracted text.

    AirWeave uses a simple len(text) < 50 check. We use:
    - char_count: raw character count
    - meaningful_ratio: ratio of alphanumeric chars to total
    - mojibake_ratio: ratio of replacement/garbled chars
    - text_to_page_ratio: chars per page (PDF-specific)
    - verdict: final assessment
    """

    char_count: int = 0
    meaningful_ratio: float = 0.0
    mojibake_ratio: float = 0.0
    text_to_page_ratio: float = 0.0
    page_count: int = 0
    verdict: QualityVerdict = QualityVerdict.EMPTY
    details: str = ""


def score_text_quality(
    text: str,
    *,
    page_count: int = 1,
    min_chars: int = 50,
    min_meaningful_ratio: float = 0.3,
    max_mojibake_ratio: float = 0.1,
    min_chars_per_page: float = 20.0,
) -> QualityScore:
    """Score extracted text quality using multiple signals.

    Args:
        text: The extracted text to score.
        page_count: Number of pages (for per-page ratio).
        min_chars: Minimum total chars to pass.
        min_meaningful_ratio: Minimum ratio of alphanumeric chars.
        max_mojibake_ratio: Maximum ratio of replacement/garbled chars.
        min_chars_per_page: Minimum chars per page for PDFs.

    Returns:
        A QualityScore with the verdict.
    """
    if not text or not text.strip():
        return QualityScore(verdict=QualityVerdict.EMPTY, details="no text")

    stripped = text.strip()
    char_count = len(stripped)

    # Count meaningful characters (letters, digits, common punctuation)
    meaningful = sum(1 for c in stripped if c.isalnum() or c in " \t\n.,;:!?-")
    meaningful_ratio = meaningful / char_count if char_count > 0 else 0.0

    # Detect mojibake / replacement characters
    mojibake_count = 0
    for c in stripped:
        if c == "\ufffd":
            mojibake_count += 1
        elif unicodedata.category(c) in ("Co", "Cn"):
            # Private use / unassigned codepoints
            mojibake_count += 1
    mojibake_ratio = mojibake_count / char_count if char_count > 0 else 0.0

    # Per-page ratio
    pages = max(1, page_count)
    text_to_page_ratio = char_count / pages

    score = QualityScore(
        char_count=char_count,
        meaningful_ratio=meaningful_ratio,
        mojibake_ratio=mojibake_ratio,
        text_to_page_ratio=text_to_page_ratio,
        page_count=page_count,
    )

    # Verdict logic
    if char_count < min_chars:
        score.verdict = QualityVerdict.POOR
        score.details = "below minimum char count (%d < %d)" % (char_count, min_chars)
    elif meaningful_ratio < min_meaningful_ratio:
        score.verdict = QualityVerdict.POOR
        score.details = "low meaningful ratio (%.2f < %.2f)" % (
            meaningful_ratio,
            min_meaningful_ratio,
        )
    elif mojibake_ratio > max_mojibake_ratio:
        score.verdict = QualityVerdict.POOR
        score.details = "high mojibake ratio (%.2f > %.2f)" % (mojibake_ratio, max_mojibake_ratio)
    elif page_count > 1 and text_to_page_ratio < min_chars_per_page:
        score.verdict = QualityVerdict.POOR
        score.details = "low chars per page (%.1f < %.1f)" % (
            text_to_page_ratio,
            min_chars_per_page,
        )
    else:
        score.verdict = QualityVerdict.GOOD
        score.details = "quality OK"

    return score


# Conversion result


@dataclass
class ConversionResult:
    """Result from a document converter.

    Attributes:
        content: Extracted markdown/text content.
        metadata: Format-specific metadata (parser name, page count, etc.).
        quality: Quality assessment of the extraction.
        needs_ocr: Whether OCR fallback is recommended.
        ocr_page_indices: For PDFs, specific page indices needing OCR
            (enables partial OCR for only the pages that need it).
    """

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: Optional[QualityScore] = None
    needs_ocr: bool = False
    ocr_page_indices: Optional[List[int]] = None


# Base converter


class BaseConverter(ABC):
    """Base class for all document converters.

    Each converter handles a specific format and converts file bytes
    to markdown text with metadata.
    """

    # Human-readable name for logging
    format_name: str = "unknown"

    @abstractmethod
    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert file bytes to markdown text.

        Args:
            file_bytes: Raw file content.
            filename: Original filename for extension detection.
            mime_type: MIME type of the file.

        Returns:
            ConversionResult with extracted text and metadata.
        """


class HybridConverter(BaseConverter):
    """Converter that tries local extraction first, then OCR fallback.

    Subclasses implement _try_extract for format-specific extraction.
    The shared convert method handles the extract-first / OCR-fallback
    orchestration, including partial OCR for PDFs (per-page, not whole-doc).

    Improvements over AirWeave's HybridDocumentConverter:
    - Quality scoring instead of simple char-count threshold
    - Partial OCR support (only OCR pages that need it)
    - Dual OCR providers (Mistral + LlamaParse)
    """

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert with extract-first, quality-score, OCR-fallback strategy."""
        try:
            result = await self._try_extract(file_bytes, filename, mime_type)
            if result.content and result.quality and result.quality.verdict == QualityVerdict.GOOD:
                logger.debug(
                    "%s: extracted via text layer (%d chars)",
                    filename,
                    len(result.content),
                )
                return result

            # Text extraction yielded poor/empty results
            if result.quality:
                logger.debug(
                    "%s: extraction quality %s (%s), recommending OCR",
                    filename,
                    result.quality.verdict.value,
                    result.quality.details,
                )
            result.needs_ocr = True
            return result

        except Exception as exc:
            logger.warning(
                "%s: extraction error (%s), recommending OCR",
                filename,
                exc,
            )
            return ConversionResult(
                needs_ocr=True,
                metadata={"parser": "local:%s" % self.format_name, "error": str(exc)},
            )

    @abstractmethod
    async def _try_extract(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Attempt local text extraction for a single file.

        Returns ConversionResult with quality scoring.
        The caller uses quality.verdict to decide OCR fallback.
        """


# Text helpers


def text_to_markdown(text: str) -> str:
    """Convert extracted plain text to basic Markdown with heuristics.

    Detects headings, bullet points, and numbered lists.
    """
    if not text:
        return ""

    lines = text.split("\n")
    result: List[str] = []
    prev_blank = True

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not prev_blank:
                result.append("")
                prev_blank = True
            continue

        prev_blank = False

        # Normalize bullet markers
        if len(stripped) >= 2 and stripped[0] in "\u2022\u00b7\u25e6" and stripped[1] == " ":
            result.append("- %s" % stripped[2:].strip())
            continue

        if len(stripped) >= 2 and stripped[0] in "*-" and stripped[1] == " ":
            result.append("- %s" % stripped[2:].strip())
            continue

        # Numbered lists pass through
        if re.match(r"^\d+[.)]\s", stripped):
            result.append(stripped)
            continue

        is_short = len(stripped) < 80
        is_uppercase = stripped.isupper() and len(stripped) > 3
        is_titlecase = stripped.istitle() and len(stripped) < 60
        ends_with_punct = stripped.endswith((".", ",", ";", ":", "?", "!"))

        if is_short and is_uppercase and not ends_with_punct:
            result.append("## %s" % stripped.title())
        elif is_short and is_titlecase and not ends_with_punct:
            result.append("## %s" % stripped)
        else:
            result.append(stripped)

    return "\n".join(result)


def render_markdown_table(rows: List[List[str]]) -> str:
    """Render rows as a markdown table with header separator."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    header = padded[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def detect_encoding(raw_bytes: bytes, sample_size: int = 100_000) -> str:
    """Detect encoding of raw bytes with chardet fallback.

    Returns the detected encoding name, defaulting to 'utf-8'.
    """
    # Try UTF-8 first
    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        import chardet

        detection = chardet.detect(raw_bytes[:sample_size])
        if detection and detection.get("confidence", 0) > 0.7:
            encoding = detection.get("encoding")
            if encoding:
                return str(encoding)
    except ImportError:
        logger.debug("chardet not available, defaulting to utf-8")

    return "utf-8"


def safe_decode(raw_bytes: bytes, max_replacement_ratio: float = 0.25) -> str:
    """Decode bytes to string with encoding detection and corruption check.

    Args:
        raw_bytes: Raw bytes to decode.
        max_replacement_ratio: Maximum allowed ratio of replacement chars.

    Returns:
        Decoded string.

    Raises:
        ValueError: If the content appears to be binary/corrupted.
    """
    if not raw_bytes:
        return ""

    encoding = detect_encoding(raw_bytes)
    try:
        text = raw_bytes.decode(encoding)
        if "\ufffd" not in text:
            return text
    except (UnicodeDecodeError, LookupError):
        pass

    # Fallback with replacement
    text = raw_bytes.decode("utf-8", errors="replace")
    replacement_count = text.count("\ufffd")

    if replacement_count > 0:
        ratio = replacement_count / len(text) if len(text) > 0 else 0
        if ratio > max_replacement_ratio or replacement_count > 5000:
            raise ValueError(
                "Content appears binary/corrupted: %d replacement chars (%.1f%%)"
                % (replacement_count, ratio * 100)
            )

    return text
