"""Local BM25 sparse embedding via FastEmbed (no API calls)."""

from __future__ import annotations

import logging
from rag_core.config.env_access import get_env as config_get_env
import threading

from fastembed import SparseTextEmbedding

from rag_core.search.text_builder import build_sparse_text
from rag_core.search.types import SparseVector

logger = logging.getLogger(__name__)

_BM25_MODEL_NAME = config_get_env(
    "SPARSE_EMBEDDING_MODEL_BM25",
    config_get_env("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25"),
)
_SPLADE_MODEL_NAME = config_get_env(
    "SPARSE_EMBEDDING_MODEL_SPLADE",
    "prithivida/Splade_PP_en_v1",
)

class FastEmbedSparseEmbedder:
    """Sparse embedder using FastEmbed with bm25 + optional SPLADE channels."""

    def __init__(
        self,
        bm25_model_name: str = _BM25_MODEL_NAME,
        splade_model_name: str = _SPLADE_MODEL_NAME,
        *,
        enable_splade: bool = True,
    ) -> None:
        self._bm25_model = SparseTextEmbedding(bm25_model_name)
        self._bm25_model_name = bm25_model_name
        self._splade_model_name = splade_model_name
        self._splade_enabled = enable_splade
        self._splade_model: SparseTextEmbedding | None = None
        self._lock = threading.Lock()

    def _embed_with_model(self, model: SparseTextEmbedding, texts: list[str]) -> list[SparseVector]:
        with self._lock:
            raw_results = list(model.embed(texts))
        return [
            SparseVector(
                indices=list(result.indices),
                values=list(result.values),
            )
            for result in raw_results
        ]

    def _ensure_splade_model(self) -> SparseTextEmbedding | None:
        if not self._splade_enabled:
            return None
        if self._splade_model is not None:
            return self._splade_model
        try:
            self._splade_model = SparseTextEmbedding(self._splade_model_name)
            logger.info("Loaded SPLADE sparse model: %s", self._splade_model_name)
            return self._splade_model
        except Exception as exc:
            self._splade_enabled = False
            logger.warning(
                "Failed to load SPLADE sparse model %s, using bm25 only: %s",
                self._splade_model_name,
                exc,
            )
            return None

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Embed multiple texts as sparse BM25 vectors."""
        return self._embed_with_model(self._bm25_model, texts)

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        """Embed texts into multiple sparse channels (bm25 + splade)."""
        bm25_vectors = self.embed_texts(texts)
        merged: list[dict[str, SparseVector]] = [{"bm25": vector} for vector in bm25_vectors]

        splade_model = self._ensure_splade_model()
        if splade_model is None:
            return merged

        splade_vectors = self._embed_with_model(splade_model, texts)
        if len(splade_vectors) != len(merged):
            logger.warning(
                "SPLADE vector count mismatch (expected=%d actual=%d); dropping splade channel",
                len(merged),
                len(splade_vectors),
            )
            return merged

        for idx, vector in enumerate(splade_vectors):
            merged[idx]["splade"] = vector
        return merged

    def embed_query(self, query: str) -> SparseVector:
        """Embed a single query as a sparse BM25 vector."""
        return self.embed_texts([query])[0]

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        """Embed a single query into all available sparse channels."""
        return self.embed_texts_multi([query])[0]
