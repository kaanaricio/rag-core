"""Contextual chunk augmentation for higher-recall embeddings."""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_MAX_SOURCE_CHARS = 6000


def _clean_markdown_for_prompt(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()
    return collapsed[:_MAX_SOURCE_CHARS]


async def generate_document_context(
    *,
    markdown: str,
    filename: str,
    model: str = 'gpt-4.1-mini',
) -> str:
    """Generate a compact 2-3 sentence document context snippet."""
    source = _clean_markdown_for_prompt(markdown)
    if not source:
        return ""

    system_prompt = (
        "Write 2-3 concise sentences that describe the document context for retrieval. "
        "Mention the document purpose, major topics, and likely terminology. "
        "Return plain text only."
    )
    user_prompt = "Filename: %s\n\nDocument excerpt:\n%s" % (filename, source)
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        response = await client.responses.create(
            model=model,
            input=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_output_tokens=180,
            temperature=0.0,
        )
        return (response.output_text or '').strip()
    except Exception:
        logger.debug("Context generation failed for %s", filename, exc_info=True)
        return ""


async def contextualize_chunks_for_embedding(
    *,
    markdown: str,
    chunks: List[str],
    filename: str,
    model: str = 'gpt-4.1-mini',
) -> List[str]:
    """Prepend shared document context to chunk text used for embeddings."""
    if not chunks:
        return []

    if len(chunks) == 1:
        return list(chunks)

    context = await generate_document_context(
        markdown=markdown,
        filename=filename,
        model=model,
    )
    if not context:
        return list(chunks)

    return ["%s\n\n%s" % (context, chunk) for chunk in chunks]
