"""XML converter — pretty-prints XML in a markdown code fence."""

from __future__ import annotations

import asyncio
import logging
from defusedxml import minidom

from .base import BaseConverter, ConversionResult, safe_decode, score_text_quality

logger = logging.getLogger(__name__)


class XmlConverter(BaseConverter):
    """Converts XML files to pretty-printed markdown code fences."""

    format_name = "xml"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert XML to pretty-printed code fence."""

        def _convert() -> ConversionResult:
            try:
                text = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": "local:xml", "error": str(exc)},
                )

            if not text.strip():
                return ConversionResult(
                    metadata={"parser": "local:xml"},
                    quality=score_text_quality(""),
                )

            try:
                dom = minidom.parseString(text)
                formatted = dom.toprettyxml(indent="  ")
                content = "```xml\n%s\n```" % formatted
            except Exception as exc:
                logger.debug("XML parsing failed for %s: %s, using raw", filename, exc)
                content = "```xml\n%s\n```" % text

            quality = score_text_quality(content)

            return ConversionResult(
                content=content,
                metadata={"parser": "local:xml", "needs_ocr": False},
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
