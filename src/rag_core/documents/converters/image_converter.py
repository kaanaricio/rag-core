"""Image converter — always requires OCR.

Images cannot be text-extracted locally, so this converter
always returns needs_ocr=True. The ingest pipeline handles
routing to Mistral or LlamaParse OCR providers.
"""

from __future__ import annotations

import logging

from .base import BaseConverter, ConversionResult, QualityScore, QualityVerdict

logger = logging.getLogger(__name__)


class ImageConverter(BaseConverter):
    """Converts image files by signaling OCR is required.

    Images (JPG, PNG, GIF, WEBP, BMP, TIFF) cannot have text
    extracted locally. This converter returns needs_ocr=True
    so the ingest pipeline can route to an OCR provider.
    """

    format_name = "image"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Signal that OCR is needed for image files."""
        return ConversionResult(
            needs_ocr=True,
            metadata={
                "parser": "ocr_required",
                "mime_type": mime_type,
                "needs_ocr": True,
            },
            quality=QualityScore(
                verdict=QualityVerdict.EMPTY,
                details="image file requires OCR",
            ),
        )
