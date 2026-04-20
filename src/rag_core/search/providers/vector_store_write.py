"""Write-path helpers for the Qdrant vector store."""

from __future__ import annotations

import asyncio
import logging
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from .vector_store_shared import (
    _MAX_SPLIT_DEPTH,
    _SLOW_WRITE_THRESHOLD_SECONDS,
    _SPLIT_PAUSE_SECONDS,
    WriteLatencyTracker,
)

logger = logging.getLogger(__name__)


def split_into_batches(
    points: list[rest.PointStruct],
    batch_size: int,
) -> list[list[rest.PointStruct]]:
    """Split points into batches of at most ``batch_size``."""
    if batch_size <= 0:
        return [points] if points else []
    return [points[i : i + batch_size] for i in range(0, len(points), batch_size)]


def log_upsert_error(
    exc: Exception,
    collection_name: str,
    dimensions: int,
    points: list[rest.PointStruct],
    split_depth: int,
) -> None:
    """Log detailed error context for a failed upsert."""
    n = len(points)
    error_context: dict[str, object] = {
        "exception_type": type(exc).__name__,
        "collection": collection_name,
        "dimensions": dimensions,
        "batch_size": n,
        "split_depth": split_depth,
        "max_split_depth": _MAX_SPLIT_DEPTH,
    }

    if isinstance(exc, UnexpectedResponse):
        error_context["http_status"] = exc.status_code
        error_context["reason"] = exc.reason_phrase

    if points:
        try:
            sample_keys = list((points[0].payload or {}).keys())[:10]
            error_context["sample_payload_keys"] = sample_keys
        except Exception:
            # Context enrichment is best-effort; the upsert failure is already logged.
            pass

    logger.error(
        "Qdrant upsert failed: %s (batch=%d, depth=%d, collection=%s). Context: %s",
        exc,
        n,
        split_depth,
        collection_name,
        error_context,
    )


async def upsert_with_fallback(
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    max_batch_size: int,
    points: list[rest.PointStruct],
    split_depth: int,
) -> None:
    """Upsert a batch, splitting in half on failure."""
    n = len(points)
    min_batch = max(1, max_batch_size // 16)

    try:
        start = time.monotonic()
        await client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        duration = time.monotonic() - start
        latency.record(duration)

        if duration > _SLOW_WRITE_THRESHOLD_SECONDS:
            logger.warning(
                "Slow Qdrant write: %d points in %.2fs "
                "(collection=%s, dims=%d, p50=%.3fs, p95=%.3fs)",
                n,
                duration,
                collection_name,
                dimensions,
                latency.p50 or 0.0,
                latency.p95 or 0.0,
            )
        else:
            logger.debug(
                "Upserted %d points in %.2fs to %s",
                n,
                duration,
                collection_name,
            )

    except (UnexpectedResponse, ResponseHandlingException) as exc:
        log_upsert_error(exc, collection_name, dimensions, points, split_depth)
        await _maybe_split_and_retry(
            client=client,
            collection_name=collection_name,
            dimensions=dimensions,
            latency=latency,
            max_batch_size=max_batch_size,
            points=points,
            split_depth=split_depth,
            min_batch=min_batch,
            original_error=exc,
        )

    except Exception as exc:
        # Catch timeout errors from httpx/httpcore.
        exc_name = type(exc).__name__.lower()
        if "timeout" in exc_name:
            log_upsert_error(exc, collection_name, dimensions, points, split_depth)
            await _maybe_split_and_retry(
                client=client,
                collection_name=collection_name,
                dimensions=dimensions,
                latency=latency,
                max_batch_size=max_batch_size,
                points=points,
                split_depth=split_depth,
                min_batch=min_batch,
                original_error=exc,
            )
        else:
            raise


async def _maybe_split_and_retry(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    max_batch_size: int,
    points: list[rest.PointStruct],
    split_depth: int,
    min_batch: int,
    original_error: Exception,
) -> None:
    n = len(points)
    if n <= min_batch or split_depth >= _MAX_SPLIT_DEPTH:
        logger.error(
            "Cannot split further: %d points, depth=%d/%d, min_batch=%d. "
            "Raising original error.",
            n,
            split_depth,
            _MAX_SPLIT_DEPTH,
            min_batch,
        )
        raise original_error

    mid = n // 2
    left, right = points[:mid], points[mid:]
    logger.warning(
        "Splitting failed batch: %d -> %d + %d (depth=%d/%d)",
        n,
        len(left),
        len(right),
        split_depth + 1,
        _MAX_SPLIT_DEPTH,
    )

    await asyncio.sleep(_SPLIT_PAUSE_SECONDS)
    await upsert_with_fallback(
        client=client,
        collection_name=collection_name,
        dimensions=dimensions,
        latency=latency,
        max_batch_size=max_batch_size,
        points=left,
        split_depth=split_depth + 1,
    )
    await asyncio.sleep(_SPLIT_PAUSE_SECONDS)
    await upsert_with_fallback(
        client=client,
        collection_name=collection_name,
        dimensions=dimensions,
        latency=latency,
        max_batch_size=max_batch_size,
        points=right,
        split_depth=split_depth + 1,
    )
