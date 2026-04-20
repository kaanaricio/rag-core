"""Build textual representations with metadata headers for embedding.

Structured metadata header + content body for document chunks.
Code files skip the metadata header (AST chunking works better on raw code).
"""

from __future__ import annotations

from typing import Any, Optional, Union

from rag_core.search.types import ContentType


def build_sparse_text(
    chunk_text: str,
    metadata: dict[str, Any],
) -> str:
    """Build plain-word sparse input from metadata and chunk content."""
    parts: list[str] = []
    for _key, value in sorted(metadata.items()):
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.extend(str(v).strip() for v in value if str(v).strip())
    parts.append(chunk_text)
    return " ".join(parts)


def build_textual_representation(
    content: str,
    source_type: str,
    name: str,
    content_type: Union[ContentType, str],
    *,
    path: Optional[str] = None,
    extra_fields: Optional[dict[str, str]] = None,
) -> str:
    """Build a textual representation with metadata header + content.

    Code files: returns raw content (no header).
    Documents: uses "# Metadata" header.
    """
    if content_type == ContentType.CODE:
        return content

    return _build_document_representation(
        content=content,
        source_type=source_type,
        name=name,
        path=path,
        extra_fields=extra_fields,
    )


def _build_document_representation(
    content: str,
    source_type: str,
    name: str,
    path: Optional[str] = None,
    extra_fields: Optional[dict[str, str]] = None,
) -> str:
    lines = [
        "# Metadata",
        "",
        f"**Source Type**: {source_type}",
        "**Type**: Document",
        f"**Name**: {name}",
    ]
    if path:
        lines.append(f"**Path**: {path}")

    if extra_fields:
        lines.append("")
        for key, value in extra_fields.items():
            lines.append(f"**{key}**: {value}")

    lines.extend(["", "# Content", "", content])
    return "\n".join(lines)
