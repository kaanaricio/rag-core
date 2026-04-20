"""Document converter registry with MIME type and extension lookup."""

from __future__ import annotations

import logging
import os

from .base import BaseConverter, ConversionResult, QualityVerdict
from .registry_loader import get_registered_converters, load_converter_class
from .registry_maps import EXTENSION_MAP, MIME_TYPE_MAP
from .registry_specs import PUBLIC_CONVERTER_CLASSES

logger = logging.getLogger(__name__)

_STATIC_EXPORTS = {
    "BaseConverter": BaseConverter,
    "ConversionResult": ConversionResult,
    "QualityVerdict": QualityVerdict,
}


def get_converter(
    *,
    mime_type: str = "",
    filename: str = "",
) -> BaseConverter:
    """Get the appropriate converter for a file.

    Resolution order:
    1. MIME type mapping
    2. File extension mapping
    3. Fallback to text converter (for text/* MIME types)
    4. Fallback to text converter (unknown types try text extraction)

    Args:
        mime_type: MIME type of the file.
        filename: Original filename for extension detection.

    Returns:
        A BaseConverter instance for the file type.
    """
    converters = get_registered_converters()
    mt = (mime_type or "").strip().lower()
    _, ext = os.path.splitext((filename or "").lower())

    key = MIME_TYPE_MAP.get(mt)
    if key and key in converters:
        return converters[key]

    key = EXTENSION_MAP.get(ext)
    if key and key in converters:
        return converters[key]

    if mt.startswith("text/"):
        return converters["text"]

    logger.debug(
        "No specific converter for mime=%s ext=%s, using text fallback",
        mt,
        ext,
    )
    return converters["text"]


async def convert_file(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> ConversionResult:
    """Convenience function to convert a file using the appropriate converter.

    Args:
        file_bytes: Raw file content.
        filename: Original filename.
        mime_type: MIME type of the file.

    Returns:
        ConversionResult with extracted text and metadata.
    """
    converter = get_converter(mime_type=mime_type, filename=filename)
    logger.debug(
        "Using %s converter for %s (mime=%s)",
        converter.format_name,
        filename,
        mime_type,
    )

    result = await converter.convert(file_bytes, filename, mime_type)

    # Log converter selection and quality
    if result.quality:
        logger.info(
            "Converted %s: converter=%s, quality=%s, chars=%d, needs_ocr=%s",
            filename,
            converter.format_name,
            result.quality.verdict.value,
            result.quality.char_count,
            result.needs_ocr,
        )
    else:
        logger.info(
            "Converted %s: converter=%s, needs_ocr=%s",
            filename,
            converter.format_name,
            result.needs_ocr,
        )

    return result


__all__ = [
    "BaseConverter",
    "ConversionResult",
    "QualityVerdict",
    "convert_file",
    "get_converter",
    *PUBLIC_CONVERTER_CLASSES,
]


def __getattr__(name: str):
    if name in _STATIC_EXPORTS:
        return _STATIC_EXPORTS[name]
    if name in {"convert_file", "get_converter"}:
        return {
            "convert_file": convert_file,
            "get_converter": get_converter,
        }[name]
    if name in PUBLIC_CONVERTER_CLASSES:
        return load_converter_class(name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
