"""PDF text hardening helpers.

These helpers target a specific failure mode in native PDF extraction:
font encodings with broken or missing Unicode maps. In those cases the page
often contains long strings of private-use characters, control bytes, or
mixed scripts that look like "text" by length alone but are unusable for
retrieval and classification.
"""

from __future__ import annotations


def strip_problematic_control_chars(text: str) -> str:
    """Remove control characters that should never survive PDF extraction."""

    result: list[str] = []
    for char in text:
        code = ord(char)
        if (0x00 <= code <= 0x1F and code not in (0x09, 0x0A, 0x0D)) or (0x80 <= code <= 0x9F):
            continue
        result.append(char)
    return "".join(result)


def looks_like_garbled_pdf_text(text: str) -> bool:
    """Return True when extracted text strongly resembles broken font output."""

    if len(text) < 3:
        return False

    private_use_count = 0
    arabic_count = 0
    latin_extended_count = 0
    basic_latin_letter_count = 0
    suspicious_count = 0
    control_char_count = 0
    normal_char_count = 0

    for char in text:
        code = ord(char)

        if (0x00 <= code <= 0x1F and code not in (0x09, 0x0A, 0x0D)) or (0x80 <= code <= 0x9F):
            control_char_count += 1
        elif 0xE000 <= code <= 0xF8FF:
            private_use_count += 1
        elif (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F) or (0x08A0 <= code <= 0x08FF):
            arabic_count += 1
        elif (0x0100 <= code <= 0x024F) or (0x1E00 <= code <= 0x1EFF):
            latin_extended_count += 1
        elif (0x41 <= code <= 0x5A) or (0x61 <= code <= 0x7A):
            basic_latin_letter_count += 1
            normal_char_count += 1
        elif (
            (0x0700 <= code <= 0x07FF)
            or (0x0800 <= code <= 0x083F)
            or (0xFFF0 <= code <= 0xFFFF)
            or (0x2500 <= code <= 0x25FF)
            or (0x0300 <= code <= 0x036F)
        ):
            suspicious_count += 1
        elif (0x20 <= code <= 0x7E) or code in (0x09, 0x0A, 0x0D):
            normal_char_count += 1

    total_chars = len(text)

    if control_char_count > 0 and control_char_count > normal_char_count:
        return True
    if private_use_count >= 2:
        return True
    if arabic_count >= 2 and latin_extended_count >= 2:
        return True
    if suspicious_count >= 3 or suspicious_count > total_chars * 0.2:
        return True
    if latin_extended_count > total_chars * 0.3 and basic_latin_letter_count < total_chars * 0.2:
        return True
    if (arabic_count >= 1 or suspicious_count >= 1) and latin_extended_count >= 3:
        return True

    return False


def normalize_pdf_extracted_text(text: str) -> tuple[str, bool]:
    """Return cleaned text and whether the original extraction looked garbled."""

    raw_text = str(text or "")
    garbled = looks_like_garbled_pdf_text(raw_text)
    cleaned = strip_problematic_control_chars(raw_text)
    return cleaned.strip(), garbled
