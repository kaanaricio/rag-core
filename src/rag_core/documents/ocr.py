from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class OcrRequest:
    file_bytes: bytes
    filename: str
    mime_type: str
    page_indices: list[int] = field(default_factory=list)
    existing_markdown: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OcrResult:
    markdown: str
    merge_mode: str = "append"
    provider_name: str | None = None
    model_name: str | None = None
    pages_processed: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OcrProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str | None: ...

    @property
    def supports_page_selection(self) -> bool: ...

    async def extract_markdown(self, request: OcrRequest) -> OcrResult: ...


class CommandOcrProvider:
    def __init__(
        self,
        *,
        command: list[str],
        provider_name: str = "command",
        model_name: str | None = None,
        supports_page_selection: bool = True,
        timeout_seconds: float = 120.0,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("CommandOcrProvider requires a non-empty command")
        self._command = list(command)
        self._provider_name = provider_name
        self._model_name = model_name
        self._supports_page_selection = supports_page_selection
        self._timeout_seconds = timeout_seconds
        self._extra_env = dict(extra_env or {})

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def supports_page_selection(self) -> bool:
        return self._supports_page_selection

    async def extract_markdown(self, request: OcrRequest) -> OcrResult:
        return _run_command_provider(
            request=request,
            command=self._command,
            provider_name=self._provider_name,
            model_name=self._model_name,
            supports_page_selection=self._supports_page_selection,
            timeout_seconds=self._timeout_seconds,
            extra_env=self._extra_env,
        )


def build_mistral_ocr_provider(
    *,
    model_name: str = "mistral-ocr-latest",
    python_executable: str | None = None,
    timeout_seconds: float = 300.0,
    extra_env: dict[str, str] | None = None,
) -> CommandOcrProvider:
    # Shape: ["/usr/bin/python3", "-m", "rag_core.documents.ocr_commands.mistral", "--model", "mistral-ocr-latest"]
    command = [
        python_executable or sys.executable,
        "-m",
        "rag_core.documents.ocr_commands.mistral",
        "--model",
        model_name,
    ]
    return CommandOcrProvider(
        command=command,
        provider_name="mistral",
        model_name=model_name,
        supports_page_selection=True,
        timeout_seconds=timeout_seconds,
        extra_env=extra_env,
    )


def build_gemini_ocr_provider(
    *,
    model_name: str = "gemini-2.5-flash",
    python_executable: str | None = None,
    timeout_seconds: float = 300.0,
    extra_env: dict[str, str] | None = None,
) -> CommandOcrProvider:
    # Shape: ["/usr/bin/python3", "-m", "rag_core.documents.ocr_commands.gemini", "--model", "gemini-2.5-flash"]
    command = [
        python_executable or sys.executable,
        "-m",
        "rag_core.documents.ocr_commands.gemini",
        "--model",
        model_name,
    ]
    return CommandOcrProvider(
        command=command,
        provider_name="gemini",
        model_name=model_name,
        supports_page_selection=False,
        timeout_seconds=timeout_seconds,
        extra_env=extra_env,
    )


def _run_command_provider(
    *,
    request: OcrRequest,
    command: list[str],
    provider_name: str,
    model_name: str | None,
    supports_page_selection: bool,
    timeout_seconds: float,
    extra_env: dict[str, str],
) -> OcrResult:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=_suffix_for_filename(request.filename),
            delete=False,
        ) as temp_file:
            temp_file.write(request.file_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        payload = {
            "file_path": temp_path,
            "filename": request.filename,
            "mime_type": request.mime_type,
            "page_indices": request.page_indices if supports_page_selection else [],
            "existing_markdown": request.existing_markdown,
            "metadata": request.metadata,
        }
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=_merge_env(extra_env),
        )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"OCR provider {provider_name} failed with code {completed.returncode}: {stderr}"
        )

    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OCR provider {provider_name} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"OCR provider {provider_name} returned a non-object payload")

    raw_markdown = parsed.get("markdown")
    if not isinstance(raw_markdown, str):
        raise RuntimeError(f"OCR provider {provider_name} returned markdown={raw_markdown!r}")

    raw_metadata = parsed.get("metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    return OcrResult(
        markdown=raw_markdown,
        merge_mode=_resolve_merge_mode(
            raw_mode=parsed.get("merge_mode"),
            supports_page_selection=supports_page_selection,
            requested_page_indices=request.page_indices,
        ),
        provider_name=str(parsed.get("provider_name") or provider_name),
        model_name=_optional_str(parsed.get("model_name")) or model_name,
        pages_processed=_normalize_page_indices(parsed.get("pages_processed", request.page_indices)),
        metadata=metadata,
    )


def _merge_env(extra_env: dict[str, str]) -> dict[str, str]:
    merged = dict(os.environ)
    merged.update(extra_env)
    return merged


def _suffix_for_filename(filename: str) -> str:
    suffix = Path(filename).suffix.strip()
    return suffix if suffix else ".bin"


def _normalize_page_indices(raw_indices: object) -> list[int]:
    if not isinstance(raw_indices, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        if not isinstance(raw_index, int) or raw_index < 0 or raw_index in seen:
            continue
        seen.add(raw_index)
        normalized.append(raw_index)
    return sorted(normalized)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_merge_mode(
    *,
    raw_mode: object,
    supports_page_selection: bool,
    requested_page_indices: list[int],
) -> str:
    if raw_mode == "replace":
        return "replace"
    if raw_mode == "append":
        return "append"
    if not supports_page_selection:
        return "replace"
    if requested_page_indices:
        return "append"
    return "replace"
