"""Shared constants and small utilities for the Qdrant vector store."""

from __future__ import annotations

from collections import deque
from typing import Optional

_DENSE_VECTOR_NAME = ""  # Qdrant default vector name
_PRIMARY_SPARSE_VECTOR_NAME = "bm25"
_SECONDARY_SPARSE_VECTOR_NAME = "splade"
_KNOWN_SPARSE_VECTOR_NAMES: frozenset[str] = frozenset(
    {_PRIMARY_SPARSE_VECTOR_NAME, _SECONDARY_SPARSE_VECTOR_NAME}
)
_PREFETCH_LIMIT = 200
_MAX_SPLIT_DEPTH = 6
_SPLIT_PAUSE_SECONDS = 0.2
_SLOW_WRITE_THRESHOLD_SECONDS = 5.0
_LATENCY_WINDOW_SIZE = 100


def compute_write_params(vector_size: int) -> tuple[int, int]:
    """Compute concurrency and batch limits for a given vector dimension."""
    if vector_size >= 3000:
        return 4, 40
    if vector_size >= 1024:
        return 8, 60
    return 16, 100


class WriteLatencyTracker:
    """Tracks P50/P95 write latencies in a fixed-size window."""

    def __init__(self, window_size: int = _LATENCY_WINDOW_SIZE) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)

    def record(self, duration_seconds: float) -> None:
        """Record a write duration sample."""
        self._samples.append(duration_seconds)

    def percentile(self, p: float) -> Optional[float]:
        """Return the p-th percentile or ``None`` when no samples exist."""
        if not self._samples:
            return None
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100.0)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50(self) -> Optional[float]:
        """Median write latency."""
        return self.percentile(50)

    @property
    def p95(self) -> Optional[float]:
        """95th percentile write latency."""
        return self.percentile(95)

    @property
    def sample_count(self) -> int:
        """Number of recorded samples."""
        return len(self._samples)
