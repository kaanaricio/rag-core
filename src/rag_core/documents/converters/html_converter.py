"""HTML converter with content extraction and cleanup.

Improvements over AirWeave:
- Strips scripts, styles, navigation, footers (main content extraction)
- Preserves table structure in markdown
- Multiple fallback strategies (markdownify > bs4 > regex)
"""

from __future__ import annotations

import asyncio
import re

from .base import BaseConverter, ConversionResult, safe_decode, score_text_quality


def _strip_non_content_html(html: str) -> str:
    """Remove scripts, styles, nav, footer, and other non-content elements."""
    # Remove script/style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove nav/footer/header elements (common boilerplate)
    for tag in ("nav", "footer", "aside"):
        html = re.sub(r"<%s[^>]*>.*?</%s>" % (tag, tag), "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    return html


def _try_markdownify(html: str) -> str | None:
    """Convert HTML to markdown using markdownify (preserves structure)."""
    try:
        from markdownify import markdownify  # type: ignore[import-not-found]

        result = markdownify(html, heading_style="ATX", strip=["img", "iframe"])
        return result.strip() if result else None
    except ImportError:
        # markdownify is optional; the converter should degrade to the next strategy.
        return None
    except Exception:
        # Malformed HTML or converter-specific failures should not abort the fallback chain.
        return None


def _try_html_to_markdown(html: str) -> str | None:
    """Convert HTML to markdown using html-to-markdown (Rust-powered)."""
    try:
        from html_to_markdown import convert  # type: ignore[import-not-found]

        result = convert(html)
        return result.strip() if result else None
    except ImportError:
        # html-to-markdown is optional; the converter should degrade to the next strategy.
        return None
    except Exception:
        # Conversion quality may vary by input, so fall through to the next extractor.
        return None


def _try_beautifulsoup(html: str) -> str | None:
    """Extract text using BeautifulSoup (fallback)."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]

        soup = BeautifulSoup(html, "html.parser")

        # Try to find main content area
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            return main.get_text("\n", strip=True)
        return soup.get_text("\n", strip=True)
    except ImportError:
        # BeautifulSoup is optional; regex stripping remains as the final fallback.
        return None
    except Exception:
        # Parser edge cases should not block best-effort text extraction.
        return None


def _regex_fallback(html: str) -> str:
    """Strip HTML tags via regex (last resort)."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class HtmlConverter(BaseConverter):
    """Converts HTML to markdown with content extraction.

    Strips non-content elements (scripts, styles, nav, footer) before
    conversion. Uses a layered fallback strategy:
    1. markdownify (best structure preservation)
    2. html-to-markdown (Rust-powered, fast)
    3. BeautifulSoup (text extraction)
    4. Regex strip (last resort)
    """

    format_name = "html"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert HTML to markdown with content extraction."""

        def _convert() -> ConversionResult:
            try:
                raw_html = safe_decode(file_bytes)
            except ValueError as exc:
                # Decode failures are returned as metadata so the ingest job can record
                # a document-level conversion error without crashing the whole pipeline.
                return ConversionResult(
                    metadata={"parser": "local:html", "error": str(exc)},
                )

            if not raw_html.strip():
                return ConversionResult(
                    metadata={"parser": "local:html"},
                    quality=score_text_quality(""),
                )

            # Strip non-content elements first
            cleaned = _strip_non_content_html(raw_html)

            # Try converters in order of quality
            parser_name = "local:html"
            content = None

            result = _try_markdownify(cleaned)
            if result:
                content = result
                parser_name = "local:markdownify"

            if not content:
                result = _try_html_to_markdown(cleaned)
                if result:
                    content = result
                    parser_name = "local:html-to-markdown"

            if not content:
                result = _try_beautifulsoup(cleaned)
                if result:
                    content = result
                    parser_name = "local:bs4"

            if not content:
                content = _regex_fallback(cleaned)
                parser_name = "local:html-regex"

            quality = score_text_quality(content or "")

            return ConversionResult(
                content=content or "",
                metadata={"parser": parser_name, "needs_ocr": False},
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
