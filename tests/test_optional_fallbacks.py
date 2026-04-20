import pytest

from rag_core.config.env_access import (
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_optional_bool,
    get_env_stripped,
)
from rag_core.documents import build_gemini_ocr_provider, build_mistral_ocr_provider
from rag_core.documents.converters import get_converter
from rag_core.search.providers.reranker import NoOpReranker, create_reranker


def test_get_converter_uses_text_fallbacks() -> None:
    assert get_converter(mime_type="text/plain", filename="notes.bin").format_name == "text"
    assert get_converter(mime_type="application/x-unknown", filename="mystery.bin").format_name == "text"


def test_create_reranker_none_has_runtime_metadata() -> None:
    reranker = create_reranker(provider="none")

    assert isinstance(reranker, NoOpReranker)
    assert getattr(reranker, "_rag_core_provider_requested") == "none"
    assert getattr(reranker, "_rag_core_provider_effective") == "none"
    assert getattr(reranker, "_rag_core_fallback_reason") is None


def test_create_reranker_missing_cohere_key_falls_back_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER_STRICT_PROVIDER", raising=False)

    reranker = create_reranker(provider="cohere")

    assert isinstance(reranker, NoOpReranker)
    assert getattr(reranker, "_rag_core_provider_requested") == "cohere"
    assert getattr(reranker, "_rag_core_provider_effective") == "none"
    assert getattr(reranker, "_rag_core_fallback_reason") == "missing_cohere_api_key"


def test_create_reranker_strict_missing_cohere_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("RERANKER_STRICT_PROVIDER", "true")

    with pytest.raises(ValueError, match="missing_cohere_api_key"):
        create_reranker(provider="cohere")


def test_create_reranker_missing_zeroentropy_key_falls_back_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEROENTROPY_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER_STRICT_PROVIDER", raising=False)

    reranker = create_reranker(provider="zeroentropy")

    assert isinstance(reranker, NoOpReranker)
    assert getattr(reranker, "_rag_core_provider_requested") == "zeroentropy"
    assert getattr(reranker, "_rag_core_provider_effective") == "none"
    assert getattr(reranker, "_rag_core_fallback_reason") == "missing_zeroentropy_api_key"


def test_create_reranker_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        create_reranker(provider="made-up")


def test_build_mistral_ocr_provider_uses_module_command() -> None:
    provider = build_mistral_ocr_provider(python_executable="/tmp/python")

    assert provider.provider_name == "mistral"
    assert provider.model_name == "mistral-ocr-latest"
    assert provider.supports_page_selection is True
    assert provider._command == [
        "/tmp/python",
        "-m",
        "rag_core.documents.ocr_commands.mistral",
        "--model",
        "mistral-ocr-latest",
    ]


def test_build_gemini_ocr_provider_uses_module_command() -> None:
    provider = build_gemini_ocr_provider(python_executable="/tmp/python")

    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.5-flash"
    assert provider.supports_page_selection is False
    assert provider._command == [
        "/tmp/python",
        "-m",
        "rag_core.documents.ocr_commands.gemini",
        "--model",
        "gemini-2.5-flash",
    ]


def test_env_helpers_fall_back_on_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT", "oops")
    monkeypatch.setenv("TEST_FLOAT", "oops")
    monkeypatch.setenv("TEST_BOOL", "maybe")
    monkeypatch.setenv("TEST_STRIPPED", "  value  ")

    assert get_env_int("TEST_INT", 7) == 7
    assert get_env_float("TEST_FLOAT", 1.5) == 1.5
    assert get_env_bool("TEST_BOOL", True) is True
    assert get_env_optional_bool("TEST_BOOL") is None
    assert get_env_stripped("TEST_STRIPPED") == "value"
