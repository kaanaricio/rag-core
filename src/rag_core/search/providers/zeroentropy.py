from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

from rag_core.search.types import RerankResult

if TYPE_CHECKING:
    import types


def _import_zeroentropy() -> "types.ModuleType":
    try:
        module = importlib.import_module("zeroentropy")
    except ImportError as exc:
        raise ImportError(
            "zeroentropy package is required for ZeroEntropy providers. "
            "Install it with: pip install zeroentropy"
        ) from exc
    return module


class ZeroEntropyEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "zembed-1",
        dimensions: int = 2560,
        api_key: str | None = None,
    ) -> None:
        zeroentropy = _import_zeroentropy()
        self._client = zeroentropy.ZeroEntropy(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "zeroentropy"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts, "document")

    async def embed_query(self, query: str) -> list[float]:
        rows = await asyncio.to_thread(self._embed_sync, [query], "query")
        return rows[0]

    def _embed_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self._client.models.embed(
            model=self._model,
            input_type=input_type,
            input=texts,
            dimensions=self._dimensions,
        )
        data = getattr(response, "data", None) or []
        return [list(getattr(row, "embedding", [])) for row in data]


class ZeroEntropyReranker:
    def __init__(
        self,
        *,
        model: str = "zerank-2",
        api_key: str | None = None,
    ) -> None:
        zeroentropy = _import_zeroentropy()
        self._client = zeroentropy.ZeroEntropy(api_key=api_key)
        self._model = model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        if not documents:
            return []
        response = await asyncio.to_thread(
            self._client.models.rerank,
            model=self._model,
            query=query,
            documents=documents,
        )
        results: list[RerankResult] = []
        for index, row in enumerate((getattr(response, "results", None) or [])[:top_k]):
            text = getattr(row, "document", None)
            score = getattr(row, "score", None)
            if not isinstance(text, str):
                continue
            if not isinstance(score, (int, float, str)):
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            source_index = documents.index(text) if text in documents else index
            if not 0 <= source_index < len(documents):
                continue
            results.append(
                RerankResult(
                    index=source_index,
                    score=numeric_score,
                    text=documents[source_index],
                )
            )
        return results
