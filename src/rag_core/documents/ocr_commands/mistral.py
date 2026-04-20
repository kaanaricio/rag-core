from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import uuid
from urllib import error, request
from typing import cast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-ocr-latest")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY is required")

    payload = json.load(__import__("sys").stdin)
    file_path = Path(str(payload["file_path"]))
    file_bytes = file_path.read_bytes()
    filename = str(payload.get("filename") or file_path.name)
    mime_type = str(payload.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/pdf")
    page_indices = _normalize_page_indices(payload.get("page_indices"))

    file_id = _upload_file(
        api_key=api_key,
        filename=filename,
        file_bytes=file_bytes,
    )
    ocr_payload = _run_ocr(
        api_key=api_key,
        model=args.model,
        file_id=file_id,
        page_indices=page_indices,
    )
    raw_pages = ocr_payload.get("pages")
    page_count = len(raw_pages) if isinstance(raw_pages, list) else 0
    markdown = _collect_markdown(raw_pages, page_indices)
    result = {
        "markdown": markdown,
        "merge_mode": "append" if page_indices else "replace",
        "provider_name": "mistral",
        "model_name": args.model,
        "pages_processed": page_indices or _default_page_indices(ocr_payload.get("pages")),
        "metadata": {
            "ocr_source_mime_type": mime_type,
            "ocr_page_count": len(page_indices) if page_indices else page_count,
            "ocr_provider_used": True,
        },
    }
    print(json.dumps(result))
    return 0


def _upload_file(*, api_key: str, filename: str, file_bytes: bytes) -> str:
    boundary = f"ragcore-{uuid.uuid4().hex}"
    body = _build_multipart_body(
        boundary=boundary,
        fields={"purpose": "ocr"},
        files={
            "file": {
                "filename": filename,
                "content_type": "application/octet-stream",
                "content": file_bytes,
            }
        },
    )
    req = request.Request(
        "https://api.mistral.ai/v1/files",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    payload = _load_json(req)
    file_id = payload.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("Mistral file upload did not return an id")
    return file_id


def _run_ocr(
    *,
    api_key: str,
    model: str,
    file_id: str,
    page_indices: list[int],
) -> dict[str, object]:
    # Shape: {"model": "mistral-ocr-latest", "document": {"file_id": "file_123"}, "pages": [0, 3]}
    payload: dict[str, object] = {
        "model": model,
        "document": {"file_id": file_id},
    }
    if page_indices:
        payload["pages"] = page_indices
    req = request.Request(
        "https://api.mistral.ai/v1/ocr",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _load_json(req)


def _load_json(req: request.Request) -> dict[str, object]:
    try:
        with request.urlopen(req) as response:
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mistral OCR request failed ({exc.code}): {body}") from exc


def _collect_markdown(raw_pages: object, requested_indices: list[int]) -> str:
    if not isinstance(raw_pages, list):
        return ""
    selected: list[str] = []
    selected_indices = set(requested_indices)
    for fallback_index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            continue
        raw_markdown = raw_page.get("markdown")
        if not isinstance(raw_markdown, str) or not raw_markdown.strip():
            continue
        page_index = fallback_index if not isinstance(raw_page.get("index"), int) else max(int(raw_page["index"]) - 1, 0)
        if selected_indices and page_index not in selected_indices:
            continue
        selected.append(raw_markdown.strip())
    return "\n\n".join(selected)


def _default_page_indices(raw_pages: object) -> list[int]:
    if not isinstance(raw_pages, list):
        return []
    return list(range(len(raw_pages)))


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
    return normalized


def _build_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    files: dict[str, dict[str, str | bytes]],
) -> bytes:
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for key, spec in files.items():
        filename = str(spec["filename"])
        content_type = str(spec["content_type"])
        content = spec["content"]
        if not isinstance(content, bytes):
            raise TypeError("multipart file content must be bytes")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
