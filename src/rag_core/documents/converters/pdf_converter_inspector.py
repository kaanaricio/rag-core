"""PDF inspector helpers for routing and metadata."""

from __future__ import annotations

import logging
from typing import Dict, List, TypeAlias

from ..pdf_inspector import PdfInspectorDetectionResult, PdfInspectorExtractionResult

logger = logging.getLogger(__name__)

_MAX_INSPECTOR_OCR_PAGE_INDICES = 400
InspectorResult: TypeAlias = PdfInspectorDetectionResult | PdfInspectorExtractionResult


def _normalize_inspector_route(route: str) -> str:
    return "".join(char for char in route.lower() if char.isalnum())


def _get_inspector_field(result: InspectorResult | None, name: str) -> object | None:
    if result is None:
        return None
    return getattr(result, name, None)


def _get_inspector_page_indices(result: InspectorResult | None, name: str) -> list[int]:
    raw_value = _get_inspector_field(result, name)
    if not isinstance(raw_value, list):
        return []
    return [value for value in raw_value if isinstance(value, int)]


def _get_inspector_metadata(result: InspectorResult | None) -> Dict[str, object]:
    return {}


def _get_inspector_page_count(*results: InspectorResult | None) -> int | None:
    for result in results:
        if result is not None and result.page_count > 0:
            return result.page_count
    return None


def _get_inspector_route(result: InspectorResult | None) -> str:
    if result is None:
        return ""
    return result.pdf_type.strip().lower()


def _inspector_is_text_based(result: InspectorResult | None) -> bool:
    route_key = _normalize_inspector_route(_get_inspector_route(result))
    return route_key in {
        "text",
        "textbased",
        "textnative",
        "nativetext",
        "digitaltext",
        "textual",
    }


def _get_inspector_markdown(result: PdfInspectorExtractionResult | None) -> str:
    if result is None:
        return ""
    return result.markdown.strip()


def _apply_inspector_analysis_metadata(
    metadata: Dict[str, object],
    *,
    detection: InspectorResult | None,
    extraction: PdfInspectorExtractionResult | None,
    ocr_page_indices: List[int],
) -> None:
    raw_route = _get_inspector_route(detection)
    if raw_route:
        metadata["inspector_route"] = raw_route

    has_encoding_issues = _get_inspector_field(extraction, "has_encoding_issues")
    if not isinstance(has_encoding_issues, bool):
        has_encoding_issues = _get_inspector_field(detection, "has_encoding_issues")
    if isinstance(has_encoding_issues, bool):
        metadata["inspector_has_encoding_issues"] = has_encoding_issues

    layout_candidates = [value for value in (extraction, detection) if value is not None]
    complex_pages = _normalize_inspector_ocr_page_indices(
        [
            page_index
            for value in layout_candidates
            for page_index in (
                _get_inspector_page_indices(value, "pages_with_tables")
                + _get_inspector_page_indices(value, "pages_with_columns")
            )
        ],
        page_count=None,
    )
    if complex_pages:
        metadata["complex_ocr_page_indices"] = [
            page_index for page_index in complex_pages if page_index in set(ocr_page_indices)
        ]

    is_complex = _get_inspector_field(extraction, "is_complex")
    if not isinstance(is_complex, bool):
        is_complex = _get_inspector_field(detection, "is_complex")
    if isinstance(is_complex, bool):
        metadata["inspector_is_complex"] = is_complex

    route_key = _normalize_inspector_route(raw_route)
    if route_key in {"imagebased", "imageheavy", "imageonly"} and ocr_page_indices:
        metadata["image_only_page_indices"] = list(ocr_page_indices)


def _inspector_supports_page_level_routing(result: InspectorResult | None) -> bool:
    route_key = _normalize_inspector_route(_get_inspector_route(result))
    return route_key in {"mixed", "mixedcontent"}


def _inspector_is_ocr_only_route(result: InspectorResult | None) -> bool:
    route_key = _normalize_inspector_route(_get_inspector_route(result))
    return route_key in {
        "scanned",
        "scannedsimple",
        "scannedcomplex",
        "imagebased",
        "imageheavy",
        "imageonly",
    }


def _normalize_inspector_ocr_page_indices(
    raw_indices: object,
    *,
    page_count: int | None,
    default_all_pages: bool = False,
) -> List[int]:
    normalized: List[int] = []
    seen: set[int] = set()

    if isinstance(raw_indices, list):
        for raw_index in raw_indices:
            if not isinstance(raw_index, int) or raw_index < 0:
                continue
            if page_count is not None and raw_index >= page_count:
                continue
            if raw_index in seen:
                continue
            seen.add(raw_index)
            normalized.append(raw_index)

    if normalized or not default_all_pages or not page_count or page_count <= 0:
        return normalized

    capped_page_count = min(page_count, _MAX_INSPECTOR_OCR_PAGE_INDICES)
    if capped_page_count < page_count:
        logger.warning(
            "Capping inspector OCR page indices from %d to %d pages",
            page_count,
            capped_page_count,
        )
    return list(range(capped_page_count))
