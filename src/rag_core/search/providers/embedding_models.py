from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    provider: str
    model: str
    default_dimensions: int
    max_dimensions: int | None
    supports_dimensions_override: bool
    allowed_dimensions: tuple[int, ...] | None = None


_MODEL_SPECS = {
    ("openai", "text-embedding-3-large"): EmbeddingModelSpec(
        provider="openai",
        model="text-embedding-3-large",
        default_dimensions=3072,
        max_dimensions=3072,
        supports_dimensions_override=True,
    ),
    ("openai", "text-embedding-3-small"): EmbeddingModelSpec(
        provider="openai",
        model="text-embedding-3-small",
        default_dimensions=1536,
        max_dimensions=1536,
        supports_dimensions_override=True,
    ),
    ("openai", "text-embedding-ada-002"): EmbeddingModelSpec(
        provider="openai",
        model="text-embedding-ada-002",
        default_dimensions=1536,
        max_dimensions=1536,
        supports_dimensions_override=False,
    ),
    ("voyage", "voyage-4-lite"): EmbeddingModelSpec(
        provider="voyage",
        model="voyage-4-lite",
        default_dimensions=1024,
        max_dimensions=2048,
        allowed_dimensions=(256, 512, 1024, 2048),
        supports_dimensions_override=True,
    ),
    ("voyage", "voyage-4"): EmbeddingModelSpec(
        provider="voyage",
        model="voyage-4",
        default_dimensions=1024,
        max_dimensions=2048,
        allowed_dimensions=(256, 512, 1024, 2048),
        supports_dimensions_override=True,
    ),
    ("voyage", "voyage-4-large"): EmbeddingModelSpec(
        provider="voyage",
        model="voyage-4-large",
        default_dimensions=1024,
        max_dimensions=2048,
        allowed_dimensions=(256, 512, 1024, 2048),
        supports_dimensions_override=True,
    ),
    ("zeroentropy", "zembed-1"): EmbeddingModelSpec(
        provider="zeroentropy",
        model="zembed-1",
        default_dimensions=2560,
        max_dimensions=2560,
        allowed_dimensions=(40, 80, 160, 320, 640, 1280, 2560),
        supports_dimensions_override=True,
    ),
}


def get_embedding_model_spec(provider: str, model: str) -> EmbeddingModelSpec | None:
    key = ((provider or "").strip().lower(), (model or "").strip())
    return _MODEL_SPECS.get(key)


def resolve_embedding_dimensions(
    *,
    provider: str,
    model: str,
    dimensions: int | None,
) -> int:
    normalized_provider = (provider or "").strip().lower()
    if dimensions is not None:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        spec = get_embedding_model_spec(normalized_provider, model)
        if spec and spec.allowed_dimensions is not None and dimensions not in spec.allowed_dimensions:
            raise ValueError(
                "embedding dimensions %d are not supported for %s/%s; choose one of %s"
                % (dimensions, normalized_provider, model, list(spec.allowed_dimensions))
            )
        if spec and spec.max_dimensions is not None and dimensions > spec.max_dimensions:
            raise ValueError(
                "embedding dimensions %d exceed max %d for %s/%s"
                % (dimensions, spec.max_dimensions, normalized_provider, model)
            )
        return dimensions

    spec = get_embedding_model_spec(normalized_provider, model)
    if spec is not None:
        return spec.default_dimensions

    msg = "embedding dimensions are required for unknown provider/model pair: %s/%s"
    raise ValueError(msg % (normalized_provider or "unknown", model or "unknown"))
