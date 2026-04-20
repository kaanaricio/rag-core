from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from urllib import error, request
from typing import cast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY is required")

    payload = json.load(__import__("sys").stdin)
    file_path = Path(str(payload["file_path"]))
    file_bytes = file_path.read_bytes()
    filename = str(payload.get("filename") or file_path.name)
    mime_type = str(payload.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/pdf")
    page_indices = _normalize_page_indices(payload.get("page_indices"))

    prompt = _build_prompt(page_indices)
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(file_bytes).decode("utf-8"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ]
    }
    req = request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent?key={api_key}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    response_payload = _load_json(req)
    markdown = _extract_text(response_payload)
    result = {
        "markdown": markdown,
        "merge_mode": "replace",
        "provider_name": "gemini",
        "model_name": args.model,
        "pages_processed": [],
        "metadata": {
            "ocr_provider_used": True,
            "ocr_processed_entire_document": True,
            "ocr_page_selection_supported": False,
            "ocr_page_indices_ignored": bool(page_indices),
            "ocr_source_mime_type": mime_type,
        },
    }
    print(json.dumps(result))
    return 0


def _build_prompt(page_indices: list[int]) -> str:
    if page_indices:
        return (
            "Convert this document to markdown. Preserve headings, lists, tables, and links. "
            "Return only markdown. Page filtering is not supported in this helper, so transcribe the document."
        )
    return (
        "Convert this document to markdown. Preserve headings, lists, tables, and links. "
        "Return only markdown with no explanation."
    )


def _load_json(req: request.Request) -> dict[str, object]:
    try:
        with request.urlopen(req) as response:
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini OCR request failed ({exc.code}): {body}") from exc


def _extract_text(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n\n".join(chunks)


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


if __name__ == "__main__":
    raise SystemExit(main())
