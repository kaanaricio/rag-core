"""Subprocess wrapper for Firecrawl's pdf-inspector CLI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Final

from rag_core.config.env_access import get_env_int, get_env_stripped

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS: Final[int] = 8_000
_DEFAULT_MAX_BYTES: Final[int] = 50 * 1024 * 1024
_WARNED_BINARY_KEYS: set[str] = set()


@dataclass(frozen=True)
class PdfInspectorDetectionResult:
    """Normalized detection result from pdf-inspector."""

    pdf_type: str
    page_count: int
    pages_needing_ocr: list[int]
    confidence: float | None
    has_encoding_issues: bool
    processing_time_ms: int | None
    is_complex: bool | None = None
    pages_with_tables: list[int] | None = None
    pages_with_columns: list[int] | None = None


@dataclass(frozen=True)
class PdfInspectorExtractionResult:
    """Normalized extraction result from pdf-inspector."""

    pdf_type: str
    page_count: int
    pages_needing_ocr: list[int]
    has_encoding_issues: bool
    processing_time_ms: int | None
    markdown: str
    is_complex: bool | None = None
    pages_with_tables: list[int] | None = None
    pages_with_columns: list[int] | None = None


def pdf_inspector_enabled() -> bool:
    """Return whether the pdf-inspector integration is enabled."""
    raw_mode = get_env_stripped("PDF_INSPECTOR_MODE", "")
    if raw_mode:
        return raw_mode.lower() not in {"disable", "disabled", "off", "false", "0"}
    return True


def detect_pdf_with_inspector(file_bytes: bytes) -> PdfInspectorDetectionResult | None:
    """Detect PDF type via pdf-inspector.

    Returns ``None`` when the integration is disabled, unavailable, or the CLI
    returns unusable output.
    """

    payload = _run_pdf_inspector(["detect-pdf", "--analyze", "--json"], file_bytes)
    if payload is None:
        return None

    try:
        (
            is_complex,
            pages_with_tables,
            pages_with_columns,
        ) = _parse_analysis_fields(payload)
        return PdfInspectorDetectionResult(
            pdf_type=_require_string(payload, "pdf_type"),
            page_count=_require_int(payload, "page_count"),
            pages_needing_ocr=_require_pages_needing_ocr(payload),
            confidence=_optional_float(payload.get("confidence")),
            has_encoding_issues=_optional_bool(payload.get("has_encoding_issues"), default=False),
            processing_time_ms=_optional_int(
                payload.get("processing_time_ms", payload.get("detection_time_ms"))
            ),
            is_complex=is_complex,
            pages_with_tables=pages_with_tables,
            pages_with_columns=pages_with_columns,
        )
    except ValueError as exc:
        logger.warning("pdf-inspector detection payload was invalid: %s", exc)
        return None


def extract_pdf_with_inspector(file_bytes: bytes) -> PdfInspectorExtractionResult | None:
    """Extract PDF markdown via pdf-inspector."""

    payload = _run_pdf_inspector(["pdf2md", "--json"], file_bytes)
    if payload is None:
        return None

    try:
        (
            is_complex,
            pages_with_tables,
            pages_with_columns,
        ) = _parse_analysis_fields(payload)
        return PdfInspectorExtractionResult(
            pdf_type=_require_string(payload, "pdf_type"),
            page_count=_require_int(payload, "page_count"),
            pages_needing_ocr=_require_pages_needing_ocr(payload),
            has_encoding_issues=_optional_bool(payload.get("has_encoding_issues"), default=False),
            processing_time_ms=_optional_int(
                payload.get("processing_time_ms", payload.get("detection_time_ms"))
            ),
            markdown=_require_markdown(payload),
            is_complex=is_complex,
            pages_with_tables=pages_with_tables,
            pages_with_columns=pages_with_columns,
        )
    except ValueError as exc:
        logger.warning("pdf-inspector extraction payload was invalid: %s", exc)
        return None


def describe_pdf_inspector_runtime() -> dict[str, object]:
    resolved_detect = _resolve_binary_path("detect-pdf")
    resolved_extract = _resolve_binary_path("pdf2md")
    return {
        "enabled": pdf_inspector_enabled(),
        "binary_path": get_env_stripped("PDF_INSPECTOR_BINARY_PATH", "") or None,
        "detect_pdf_available": resolved_detect is not None,
        "pdf2md_available": resolved_extract is not None,
        "timeout_ms": max(1, get_env_int("PDF_INSPECTOR_TIMEOUT_MS", _DEFAULT_TIMEOUT_MS)),
        "max_bytes": max(1, get_env_int("PDF_INSPECTOR_MAX_BYTES", _DEFAULT_MAX_BYTES)),
    }


def _run_pdf_inspector(command: list[str], file_bytes: bytes) -> dict[str, object] | None:
    if not pdf_inspector_enabled():
        return None

    max_bytes = max(1, get_env_int("PDF_INSPECTOR_MAX_BYTES", _DEFAULT_MAX_BYTES))
    if len(file_bytes) > max_bytes:
        logger.warning(
            "Skipping pdf-inspector for oversized PDF (%d bytes > %d byte limit)",
            len(file_bytes),
            max_bytes,
        )
        return None

    binary_name = command[0]
    binary_path = _resolve_binary_path(binary_name)
    if binary_path is None:
        _warn_missing_binary(binary_name)
        return None

    timeout_ms = max(1, get_env_int("PDF_INSPECTOR_TIMEOUT_MS", _DEFAULT_TIMEOUT_MS))
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        completed = subprocess.run(
            [binary_path, temp_path, *command[1:]],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_ms / 1000,
        )
    except FileNotFoundError:
        _warn_missing_binary(binary_name)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdf-inspector %s timed out after %dms", binary_name, timeout_ms)
        return None
    except OSError as exc:
        logger.warning("pdf-inspector %s failed to start: %s", binary_name, exc)
        return None
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        logger.warning(
            "pdf-inspector %s exited with code %d: %s",
            binary_name,
            completed.returncode,
            _truncate_output(completed.stderr or completed.stdout),
        )
        return None

    stdout = completed.stdout.strip()
    if not stdout:
        logger.warning("pdf-inspector %s returned empty output", binary_name)
        return None

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.warning("pdf-inspector %s returned invalid JSON: %s", binary_name, exc)
        return None

    if not isinstance(parsed, dict):
        logger.warning("pdf-inspector %s returned non-object JSON", binary_name)
        return None
    return parsed


def _resolve_binary_path(binary_name: str) -> str | None:
    configured_path = get_env_stripped("PDF_INSPECTOR_BINARY_PATH", "")
    if configured_path:
        configured = Path(configured_path)
        if configured.is_dir():
            candidate = configured / binary_name
        elif configured.name == binary_name:
            candidate = configured
        else:
            candidate = configured.parent / binary_name

        if candidate.is_file():
            return str(candidate)
        return None

    return shutil.which(binary_name)


def _warn_missing_binary(binary_name: str) -> None:
    warning_key = f"{binary_name}:{get_env_stripped('PDF_INSPECTOR_BINARY_PATH', '')}"
    if warning_key in _WARNED_BINARY_KEYS:
        return
    _WARNED_BINARY_KEYS.add(warning_key)
    logger.warning(
        "pdf-inspector binary %s was not found (PDF_INSPECTOR_BINARY_PATH=%r)",
        binary_name,
        get_env_stripped("PDF_INSPECTOR_BINARY_PATH", ""),
    )


def _require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_markdown(payload: dict[str, object]) -> str:
    value = payload.get("markdown")
    if not isinstance(value, str):
        raise ValueError("markdown must be a string")
    return value


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    coerced = _coerce_int(value)
    if coerced is None:
        raise ValueError(f"{key} must be an integer")
    return coerced


def _optional_int(value: object) -> int | None:
    return _coerce_int(value)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_analysis_fields(
    payload: dict[str, object],
) -> tuple[bool | None, list[int] | None, list[int] | None]:
    pages_with_tables = _optional_page_indices(payload.get("pages_with_tables"))
    pages_with_columns = _optional_page_indices(payload.get("pages_with_columns"))
    is_complex = _optional_nullable_bool(payload.get("is_complex"))

    if is_complex is None and pages_with_tables is not None and pages_with_columns is not None:
        is_complex = bool(pages_with_tables or pages_with_columns)

    return is_complex, pages_with_tables, pages_with_columns


def _optional_nullable_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _require_pages_needing_ocr(payload: dict[str, object]) -> list[int]:
    raw_pages = payload.get("pages_needing_ocr", [])
    if not isinstance(raw_pages, list):
        raise ValueError("pages_needing_ocr must be a list")

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_page in raw_pages:
        page_number = _coerce_int(raw_page)
        if page_number is None or page_number <= 0:
            raise ValueError("pages_needing_ocr entries must be positive integers")
        page_index = page_number - 1
        if page_index in seen:
            continue
        seen.add(page_index)
        normalized.append(page_index)
    return normalized


def _optional_page_indices(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_page in value:
        page_number = _coerce_int(raw_page)
        if page_number is None or page_number <= 0:
            continue
        page_index = page_number - 1
        if page_index in seen:
            continue
        seen.add(page_index)
        normalized.append(page_index)
    return normalized


def _truncate_output(output: str, *, limit: int = 500) -> str:
    collapsed = " ".join(output.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."
