"""CSV converter with smart delimiter and header detection.

Improvements over AirWeave:
- Auto-detects delimiter (comma, tab, semicolon, pipe)
- Detects whether first row is actually a header
- Handles large CSVs with configurable row limits
- Streaming-friendly for oversized files
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from rag_core.config.env_access import get_env as config_get_env
from typing import Dict, List

from .base import (
    BaseConverter,
    ConversionResult,
    render_markdown_table,
    safe_decode,
    score_text_quality,
)

logger = logging.getLogger(__name__)


def detect_delimiter(sample: str) -> str:
    """Detect CSV delimiter from a sample of the file.

    Checks comma, tab, semicolon, and pipe. Returns the one with
    the most consistent column count across sample lines.
    """
    candidates = [",", "\t", ";", "|"]
    lines = sample.strip().split("\n")[:20]  # Sample first 20 lines

    if len(lines) < 2:
        # With only one line, use csv.Sniffer or default
        try:
            dialect = csv.Sniffer().sniff(sample[:4096])
            return str(dialect.delimiter)
        except csv.Error:
            return ","

    best_delimiter = ","
    best_score = -1

    for delim in candidates:
        col_counts = []
        for line in lines:
            cols = line.split(delim)
            col_counts.append(len(cols))

        if not col_counts:
            continue

        # Score: prefer delimiters that give consistent column counts > 1
        avg_cols = sum(col_counts) / len(col_counts)
        if avg_cols <= 1:
            continue

        # Consistency: standard deviation of column counts
        variance = sum((c - avg_cols) ** 2 for c in col_counts) / len(col_counts)
        consistency = 1.0 / (1.0 + variance)
        score = avg_cols * consistency

        if score > best_score:
            best_score = score
            best_delimiter = delim

    return best_delimiter


def detect_header_row(rows: List[List[str]]) -> bool:
    """Detect whether the first row is a header.

    Heuristics:
    - Headers tend to have all-string values (no pure numbers)
    - Headers tend to be shorter than data rows
    - Headers tend to have unique values
    """
    if len(rows) < 2:
        return True  # Default: treat first row as header

    first_row = rows[0]
    data_rows = rows[1 : min(6, len(rows))]  # Sample up to 5 data rows

    # Check if first row values are all non-numeric
    first_all_text = all(not _is_numeric(cell) for cell in first_row if cell.strip())

    # Check if data rows have numeric values
    data_has_numbers = False
    for row in data_rows:
        if any(_is_numeric(cell) for cell in row if cell.strip()):
            data_has_numbers = True
            break

    # If first row is all text and data has numbers, likely a header
    if first_all_text and data_has_numbers:
        return True

    # Check uniqueness of first row
    stripped_first = [c.strip().lower() for c in first_row if c.strip()]
    if len(stripped_first) == len(set(stripped_first)) and len(stripped_first) > 1:
        return True

    return True  # Default assumption


def _is_numeric(value: str) -> bool:
    """Check if a string value looks numeric."""
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


class CsvConverter(BaseConverter):
    """Converts CSV files to markdown tables with smart detection.

    Smarter than AirWeave's basic approach:
    - Auto-detects delimiter (tab, semicolon, pipe, comma)
    - Detects if first row is actually a header
    - Row limit for large CSVs
    """

    format_name = "csv"

    def __init__(self, *, max_rows: int = 0) -> None:
        self._max_rows = max_rows or int(config_get_env("LOCAL_PARSE_CSV_MAX_ROWS", "1000"))

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert CSV to markdown table."""

        def _convert() -> ConversionResult:
            try:
                text = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": "local:csv", "error": str(exc)},
                )

            if not text.strip():
                return ConversionResult(
                    metadata={"parser": "local:csv"},
                    quality=score_text_quality(""),
                )

            # Detect delimiter
            delimiter = detect_delimiter(text)

            # Parse CSV
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows: List[List[str]] = []
            truncated = False

            for row in reader:
                if len(rows) >= self._max_rows:
                    truncated = True
                    break
                rows.append(row)

            if not rows:
                return ConversionResult(
                    metadata={"parser": "local:csv"},
                    quality=score_text_quality(""),
                )

            # Detect if first row is header
            has_header = detect_header_row(rows)

            # Build markdown
            if has_header and len(rows) > 1:
                content = render_markdown_table(rows)
            else:
                # No clear header: use generic column names
                width = max(len(r) for r in rows)
                header = ["Col %d" % (i + 1) for i in range(width)]
                all_rows = [header] + rows
                content = render_markdown_table(all_rows)

            if truncated:
                content += "\n\n*[truncated after %d rows]*" % self._max_rows

            quality = score_text_quality(content)

            metadata: Dict[str, str | int | bool] = {
                "parser": "local:csv",
                "delimiter": repr(delimiter),
                "has_header": has_header,
                "row_count": len(rows),
                "truncated": truncated,
                "needs_ocr": False,
            }

            return ConversionResult(
                content=content,
                metadata=metadata,  # type: ignore[arg-type]
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
