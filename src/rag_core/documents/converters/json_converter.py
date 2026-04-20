"""JSON converter — pretty-prints JSON in a markdown code fence."""

from __future__ import annotations

import asyncio
import json
import logging

from .base import BaseConverter, ConversionResult, safe_decode, score_text_quality

logger = logging.getLogger(__name__)


class JsonConverter(BaseConverter):
    """Converts JSON files to pretty-printed markdown code fences."""

    format_name = "json"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert JSON to pretty-printed code fence."""

        def _convert() -> ConversionResult:
            try:
                text = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": "local:json", "error": str(exc)},
                )

            if not text.strip():
                return ConversionResult(
                    metadata={"parser": "local:json"},
                    quality=score_text_quality(""),
                )

            try:
                data = json.loads(text)
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                content = "```json\n%s\n```" % formatted
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON in %s: %s", filename, exc)
                # Return raw content in a code fence as fallback
                content = "```json\n%s\n```" % text
                return ConversionResult(
                    content=content,
                    metadata={"parser": "local:json", "parse_error": str(exc)},
                    quality=score_text_quality(content),
                )

            quality = score_text_quality(content)

            return ConversionResult(
                content=content,
                metadata={"parser": "local:json", "needs_ocr": False},
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
