from __future__ import annotations

import re
from typing import Any

from rag_core.documents.pdf_inspector import describe_pdf_inspector_runtime


def resolve_collection_name(
    *,
    base_name: str,
    model_name: str,
    dimensions: int,
    dimension_aware: bool,
) -> str:
    if not dimension_aware:
        return base_name
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.strip().lower()).strip("_")
    return f"{base_name}__{model_slug}_{dimensions}d"


def build_runtime_description(
    *,
    collection_name: str | None,
    embedding_provider: Any,
    sparse_embedder: Any,
    reranker: Any,
    ocr_provider: Any,
) -> dict[str, object]:
    return {
        "collection_name": collection_name,
        "embedding": {
            "provider": _provider_name(embedding_provider),
            "model": getattr(embedding_provider, "model_name", None),
            "dimensions": getattr(embedding_provider, "dimensions", None),
        },
        "sparse": {
            "provider": _provider_name(sparse_embedder),
        },
        "reranker": {
            "provider": _provider_name(reranker),
            "requested": getattr(reranker, "_rag_core_provider_requested", None),
            "effective": getattr(reranker, "_rag_core_provider_effective", None),
            "fallback_reason": getattr(reranker, "_rag_core_fallback_reason", None),
        },
        "ocr": (
            {
                "provider": _provider_name(ocr_provider),
                "model": getattr(ocr_provider, "model_name", None),
                "supports_page_selection": getattr(ocr_provider, "supports_page_selection", False),
            }
            if ocr_provider is not None
            else None
        ),
        "pdf_inspector": describe_pdf_inspector_runtime(),
    }


def _provider_name(provider: Any) -> str:
    explicit = getattr(provider, "provider_name", None)
    if explicit:
        return str(explicit)
    return type(provider).__name__
