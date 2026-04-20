"""Dense embedding providers with dimension-aware defaults."""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from .embedding_models import resolve_embedding_dimensions
from .voyage import VoyageEmbeddingProvider
from .zeroentropy import ZeroEntropyEmbeddingProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with optional base URL override."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._provider = "openai"
        self._model = model
        self._dimensions = resolve_embedding_dimensions(
            provider=self._provider,
            model=model,
            dimensions=dimensions,
        )
        if api_key and base_url:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        elif api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        elif base_url:
            self._client = AsyncOpenAI(base_url=base_url)
        else:
            self._client = AsyncOpenAI()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, batching to avoid API limits."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            all_embeddings.extend(
                [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
            )
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]

def create_embedding_provider(
    *,
    provider: str = "openai",
    model: str = "text-embedding-3-large",
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAIEmbeddingProvider | VoyageEmbeddingProvider | ZeroEntropyEmbeddingProvider:
    requested = (provider or "openai").strip().lower()
    if requested == "openai":
        return OpenAIEmbeddingProvider(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            base_url=base_url,
        )
    if requested == "voyage":
        return VoyageEmbeddingProvider(
            model=model,
            dimensions=resolve_embedding_dimensions(
                provider=requested,
                model=model,
                dimensions=dimensions,
            ),
            api_key=api_key,
        )
    if requested == "zeroentropy":
        return ZeroEntropyEmbeddingProvider(
            model=model,
            dimensions=resolve_embedding_dimensions(
                provider=requested,
                model=model,
                dimensions=dimensions,
            ),
            api_key=api_key,
        )
    raise ValueError("Unknown embedding provider: %s" % requested)
