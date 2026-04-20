"""Local document parsing entrypoint for the converter-based parse path."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _is_pdf_document(*, filename: str, mime_type: str) -> bool:
    mt = (mime_type or "").strip().lower()
    if mt == "application/pdf":
        return True
    return filename.strip().lower().endswith(".pdf")


def _normalize_ocr_page_indices(raw_indices: Any) -> List[int]:
    if not isinstance(raw_indices, list):
        return []

    normalized: List[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        if not isinstance(raw_index, int) or raw_index < 0:
            continue
        if raw_index in seen:
            continue
        seen.add(raw_index)
        normalized.append(raw_index)
    normalized.sort()
    return normalized


def _allows_empty_ocr_only_output(
    *,
    filename: str,
    mime_type: str,
    metadata: Dict[str, Any],
) -> bool:
    if not _is_pdf_document(filename=filename, mime_type=mime_type):
        return False
    if not bool(metadata.get("needs_ocr")):
        return False
    return bool(_normalize_ocr_page_indices(metadata.get("ocr_page_indices")))


class LocalParseError(RuntimeError):
    """Raised when local parsing fails."""




async def parse_file_bytes(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> Tuple[str, Dict[str, Any]]:
    """Parse file using the new converter system.

    Returns (markdown, metadata) compatible with the existing pipeline.
    The metadata dict contains "parser", "needs_ocr", and optionally
    "ocr_page_indices" for partial PDF OCR.
    """
    try:
        from .converters import convert_file

        result = await convert_file(file_bytes, filename, mime_type)
        metadata: Dict[str, Any] = dict(result.metadata) if result.metadata else {}
        metadata.setdefault("parser", "local:converter")
        metadata.setdefault("needs_ocr", result.needs_ocr)

        if result.ocr_page_indices:
            metadata["ocr_page_indices"] = result.ocr_page_indices

        normalized_ocr_page_indices = _normalize_ocr_page_indices(metadata.get("ocr_page_indices"))
        if normalized_ocr_page_indices:
            metadata["ocr_page_indices"] = normalized_ocr_page_indices

        # Converter output is now the canonical parse path.
        content = result.content or ""
        if not content.strip() and not _allows_empty_ocr_only_output(
            filename=filename,
            mime_type=mime_type,
            metadata=metadata,
        ):
            raise LocalParseError(
                "Converter returned empty output for %s (parser=%s)"
                % (filename, metadata.get("parser", "unknown"))
            )

        if result.quality:
            metadata["quality_verdict"] = result.quality.verdict.value
            metadata["quality_details"] = result.quality.details

        return content, metadata

    except Exception as exc:
        logger.error("Converter system failed for %s: %s", filename, exc)
        raise LocalParseError("Converter parse failed for %s: %s" % (filename, exc)) from exc
