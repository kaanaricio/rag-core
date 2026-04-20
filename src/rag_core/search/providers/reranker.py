"""Pluggable reranking providers."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, TypeVar, cast

from rag_core.config.env_access import get_env as config_get_env
from rag_core.search.types import RerankResult, RerankerProvider

logger = logging.getLogger(__name__)
_R = TypeVar("_R", bound=object)


def _safe_rerank_results(
    *,
    rows: list[tuple[object, object]],
    documents: list[str],
    provider_name: str,
) -> list[RerankResult]:
    results: list[RerankResult] = []
    for raw_index, raw_score in rows:
        if not isinstance(raw_index, int) or not 0 <= raw_index < len(documents):
            logger.warning("%s returned invalid rerank index: %r", provider_name, raw_index)
            continue
        if not isinstance(raw_score, (int, float, str)):
            logger.warning("%s returned invalid rerank score: %r", provider_name, raw_score)
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning("%s returned invalid rerank score: %r", provider_name, raw_score)
            continue
        if not math.isfinite(score):
            logger.warning("%s returned non-finite rerank score: %r", provider_name, raw_score)
            continue
        results.append(
            RerankResult(
                index=raw_index,
                score=score,
                text=documents[raw_index],
            )
        )
    return results


class NoOpReranker:
    """Passthrough reranker that returns results in original order."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        return [
            RerankResult(index=i, score=1.0 - (i * 0.01), text=doc)
            for i, doc in enumerate(documents[:top_k])
        ]


def _import_cohere() -> Any:
    """Lazy-import cohere so the module works even when the SDK is absent."""

    try:
        import cohere
    except ImportError as exc:
        raise ImportError(
            "cohere package is required for CohereReranker. " "Install it with: pip install cohere"
        ) from exc
    if cohere is None or not hasattr(cohere, "AsyncClientV2"):
        raise ImportError("cohere package with AsyncClientV2 is required for CohereReranker.")
    return cohere


class CohereReranker:
    """Cohere rerank-v3.5 provider."""

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: Optional[str] = None,
    ) -> None:
        cohere = _import_cohere()
        self._model = model
        self._client = cohere.AsyncClientV2(api_key=api_key)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        if not documents:
            return []
        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k,
        )
        return _safe_rerank_results(
            rows=[
                (getattr(r, "index", None), getattr(r, "relevance_score", None))
                for r in getattr(response, "results", []) or []
            ],
            documents=documents,
            provider_name="CohereReranker",
        )


def _import_voyage_reranker():
    from .voyage import VoyageReranker

    return VoyageReranker


def _import_zeroentropy_reranker():
    from .zeroentropy import ZeroEntropyReranker

    return ZeroEntropyReranker


def _env_bool(name: str, default: bool) -> bool:
    raw = config_get_env(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_reranker_provider(
    provider: str,
    api_key: Optional[str] = None,
) -> tuple[str, str | None]:
    """Resolve requested provider to an effective provider.

    Returns:
        (effective_provider, fallback_reason)
    """
    requested = (provider or "none").strip().lower()
    if requested == "none":
        return "none", None

    if requested == "cohere":
        key = (api_key or config_get_env("COHERE_API_KEY") or "").strip()
        if key:
            return "cohere", None
        return "none", "missing_cohere_api_key"
    if requested == "voyage":
        key = (api_key or config_get_env("VOYAGE_API_KEY") or "").strip()
        if key:
            return "voyage", None
        return "none", "missing_voyage_api_key"
    if requested == "zeroentropy":
        key = (api_key or config_get_env("ZEROENTROPY_API_KEY") or "").strip()
        if key:
            return "zeroentropy", None
        return "none", "missing_zeroentropy_api_key"

    return "invalid", f"unknown_provider:{requested}"


def _attach_runtime_metadata(
    reranker: _R,
    *,
    requested: str,
    effective: str,
    fallback_reason: str | None,
) -> _R:
    setattr(reranker, "_rag_core_provider_requested", requested)
    setattr(reranker, "_rag_core_provider_effective", effective)
    setattr(reranker, "_rag_core_fallback_reason", fallback_reason)
    return reranker


def create_reranker(
    provider: str = "none",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RerankerProvider:
    """Factory function for creating reranker instances."""
    requested = (provider or "none").strip().lower()
    strict = _env_bool("RERANKER_STRICT_PROVIDER", False)
    effective, fallback_reason = resolve_reranker_provider(requested, api_key=api_key)
    if effective == "invalid":
        msg = "Unknown reranker provider: %s"
        raise ValueError(msg % requested)

    if fallback_reason:
        message = "Reranker provider '%s' unavailable (%s); falling back to no-op reranker." % (
            requested,
            fallback_reason,
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    if effective == "none":
        return _attach_runtime_metadata(
            NoOpReranker(),
            requested=requested,
            effective=effective,
            fallback_reason=fallback_reason,
        )

    if effective == "cohere":
        try:
            reranker = CohereReranker(
                model=model or "rerank-v3.5",
                api_key=api_key,
            )
            return _attach_runtime_metadata(
                reranker,
                requested=requested,
                effective=effective,
                fallback_reason=fallback_reason,
            )
        except Exception as exc:
            message = "Failed to initialize Cohere reranker; using no-op reranker: %s" % exc
            if strict:
                raise ValueError(message) from exc
            logger.warning(message)
            return _attach_runtime_metadata(
                NoOpReranker(),
                requested=requested,
                effective="none",
                fallback_reason="cohere_init_failed",
            )

    if effective == "voyage":
        try:
            VoyageReranker = _import_voyage_reranker()
            reranker = VoyageReranker(
                model=model or "rerank-2.5-lite",
                api_key=api_key,
            )
            return cast(
                RerankerProvider,
                _attach_runtime_metadata(
                    reranker,
                    requested=requested,
                    effective=effective,
                    fallback_reason=fallback_reason,
                ),
            )
        except Exception as exc:
            message = "Failed to initialize Voyage reranker; using no-op reranker: %s" % exc
            if strict:
                raise ValueError(message) from exc
            logger.warning(message)
            return _attach_runtime_metadata(
                NoOpReranker(),
                requested=requested,
                effective="none",
                fallback_reason="voyage_init_failed",
            )

    if effective == "zeroentropy":
        try:
            ZeroEntropyReranker = _import_zeroentropy_reranker()
            reranker = ZeroEntropyReranker(
                model=model or "zerank-2",
                api_key=api_key,
            )
            return cast(
                RerankerProvider,
                _attach_runtime_metadata(
                    reranker,
                    requested=requested,
                    effective=effective,
                    fallback_reason=fallback_reason,
                ),
            )
        except Exception as exc:
            message = "Failed to initialize ZeroEntropy reranker; using no-op reranker: %s" % exc
            if strict:
                raise ValueError(message) from exc
            logger.warning(message)
            return _attach_runtime_metadata(
                NoOpReranker(),
                requested=requested,
                effective="none",
                fallback_reason="zeroentropy_init_failed",
            )

    msg = "Unknown reranker provider: %s"
    raise ValueError(msg % provider)
