from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import cast

from rag_core.search.types import RerankResult

if TYPE_CHECKING:
    import types


def _import_voyageai() -> "types.ModuleType":
    try:
        import voyageai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "voyageai package is required for Voyage providers. Install it with: pip install voyageai"
        ) from exc
    return cast("types.ModuleType", voyageai)


class VoyageEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "voyage-4",
        dimensions: int = 1024,
        api_key: str | None = None,
    ) -> None:
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=api_key)
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
        return "voyage"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts, "document")

    async def embed_query(self, query: str) -> list[float]:
        rows = await asyncio.to_thread(self._embed_sync, [query], "query")
        return rows[0]

    def _embed_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self._client.embed(
            texts,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )
        return [list(row) for row in getattr(response, "embeddings", [])]


class VoyageReranker:
    def __init__(
        self,
        *,
        model: str = "rerank-2.5-lite",
        api_key: str | None = None,
    ) -> None:
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=api_key)
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
            self._client.rerank,
            query,
            documents,
            self._model,
            top_k,
        )
        results: list[RerankResult] = []
        for row in getattr(response, "results", []) or []:
            raw_index = getattr(row, "index", None)
            raw_score = getattr(row, "relevance_score", None)
            if not isinstance(raw_index, int) or not 0 <= raw_index < len(documents):
                continue
            if not isinstance(raw_score, (int, float, str)):
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            results.append(
                RerankResult(
                    index=raw_index,
                    score=score,
                    text=documents[raw_index],
                )
            )
        return results
