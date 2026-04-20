"""Text file converter with encoding detection.

Handles TXT, MD, YAML, TOML, and other plain text formats.
"""

from __future__ import annotations

import asyncio
import logging

from .base import BaseConverter, ConversionResult, safe_decode, score_text_quality

logger = logging.getLogger(__name__)


class TextConverter(BaseConverter):
    """Converts plain text files (TXT, MD, YAML, TOML, etc.) with encoding detection."""

    format_name = "text"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert text file with encoding detection."""

        def _convert() -> ConversionResult:
            try:
                content = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": "local:text", "error": str(exc)},
                )

            quality = score_text_quality(content)

            return ConversionResult(
                content=content,
                metadata={"parser": "local:text", "needs_ocr": False},
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
