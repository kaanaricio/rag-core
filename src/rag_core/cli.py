from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import json
import mimetypes
from pathlib import Path
from typing import Any, Sequence

from rag_core.config.env_access import get_env, get_env_bool, get_env_int
from rag_core.core_file_io import read_file_bytes
from rag_core.core_manifest import build_manifest_entry, build_preview_document
from rag_core.core_models import RAGCoreConfig
from rag_core.core_prepare import prepare_document_bytes
from rag_core.core_runtime import resolve_collection_name
from rag_core.documents.pdf_inspector import describe_pdf_inspector_runtime
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions
from rag_core.search.providers.reranker import resolve_reranker_provider
from rag_core.search.providers.vector_store import QdrantVectorStore


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "doctor":
            return await _run_doctor(args)
        if args.command == "demo":
            return await _run_demo(args)
        if args.command == "manifest":
            return await _run_manifest(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"rag-core: error: {_cli_error_message(exc)}\n")
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-core")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser(
        "doctor",
        help="Print the planned runtime shape and optionally verify the Qdrant collection.",
    )
    _add_config_flags(doctor)
    doctor.add_argument(
        "--check-store",
        action="store_true",
        help="Create/check the configured Qdrant collection and include health data.",
    )
    doctor.add_argument("--json", action="store_true", help="Emit JSON output.")
    doctor.description = (
        "Inspect collection/provider shape. This command reports config-level runtime "
        "details, not every programmatic RAGCoreConfig field."
    )

    demo = subparsers.add_parser(
        "demo",
        help="Run the built-in local demo without vendor API keys or external Qdrant.",
    )
    demo.add_argument("--json", action="store_true", help="Emit JSON output.")

    manifest = subparsers.add_parser(
        "manifest",
        help="Preview the manifest entry for one file without indexing it.",
    )
    manifest.add_argument("path", help="Path to the local file.")
    manifest.add_argument("--namespace", required=True)
    manifest.add_argument("--corpus-id", required=True)
    manifest.add_argument("--document-id")
    manifest.add_argument("--document-key")
    manifest.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable metadata field.",
    )
    manifest.add_argument("--json", action="store_true", help="Emit JSON output.")

    return parser


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qdrant-url", default=_env_or_none("RAG_CORE_QDRANT_URL"))
    parser.add_argument(
        "--qdrant-location",
        default=_env_or_none("RAG_CORE_QDRANT_LOCATION"),
    )
    parser.add_argument("--qdrant-api-key", default=_env_or_none("RAG_CORE_QDRANT_API_KEY"))
    parser.add_argument(
        "--qdrant-collection",
        default=_env_or_default("RAG_CORE_QDRANT_COLLECTION", "rag_core_chunks"),
    )
    parser.add_argument(
        "--embedding-provider",
        default=_env_or_default("RAG_CORE_EMBEDDING_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--embedding-model",
        default=_env_or_default("RAG_CORE_EMBEDDING_MODEL", "text-embedding-3-large"),
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=_env_or_int("RAG_CORE_EMBEDDING_DIMENSIONS"),
    )
    parser.add_argument("--reranker-provider", default=_env_or_default("RAG_CORE_RERANKER_PROVIDER", "none"))
    parser.add_argument("--reranker-model", default=_env_or_none("RAG_CORE_RERANKER_MODEL"))
    parser.add_argument(
        "--dimension-aware-collection",
        action=argparse.BooleanOptionalAction,
        default=get_env_bool("RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION", True),
    )


async def _run_doctor(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    payload = await _planned_runtime_payload(config, check_store=args.check_store)
    _emit_doctor(payload, as_json=args.json)
    return 0


async def _run_demo(args: argparse.Namespace) -> int:
    from rag_core.demo import run_demo_app

    payload = await run_demo_app()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"Indexed document: {payload['document_id']} ({payload['chunk_count']} chunks)")
    print("Top hits:")
    for raw_hit in payload["hits"]:
        score = raw_hit["score"]
        title = raw_hit["title"]
        text = raw_hit["text"]
        print(f"- {score:.3f} {title}: {text[:80]}")
    return 0


async def _run_manifest(args: argparse.Namespace) -> int:
    file_path = Path(args.path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    file_bytes = await read_file_bytes(file_path)
    prepared = await prepare_document_bytes(
        file_bytes=file_bytes,
        filename=file_path.name,
        mime_type=mime_type or "application/octet-stream",
        path=str(file_path),
        contextualize=False,
        ocr_provider=None,
    )
    preview = build_preview_document(
        file_bytes=file_bytes,
        prepared=prepared,
        namespace=args.namespace,
        corpus_id=args.corpus_id,
        document_id=args.document_id,
        document_key=args.document_key,
        metadata=_parse_metadata_fields(args.metadata),
    )
    entry = build_manifest_entry(preview)
    payload: dict[str, object] = {
        "document": _dataclass_payload(preview),
        "manifest_entry": _dataclass_payload(entry),
    }
    _emit_manifest(payload, as_json=args.json)
    return 0


async def _planned_runtime_payload(
    config: RAGCoreConfig,
    *,
    check_store: bool,
) -> dict[str, object]:
    dimensions = resolve_embedding_dimensions(
        provider=config.embedding_provider,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
    requested, fallback_reason = resolve_reranker_provider(
        config.reranker_provider,
        api_key=config.reranker_api_key,
    )
    collection_name = resolve_collection_name(
        base_name=config.qdrant_collection,
        model_name=config.embedding_model,
        dimensions=dimensions,
        dimension_aware=config.qdrant_dimension_aware_collection,
    )
    payload: dict[str, object] = {
        "collection_name": collection_name,
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "dimensions": dimensions,
        },
        "sparse": {
            "provider": "fastembed",
        },
        "reranker": {
            "provider": config.reranker_provider,
            "requested": config.reranker_provider,
            "effective": requested,
            "fallback_reason": fallback_reason,
            "model": config.reranker_model,
        },
        "qdrant": {
            "url": config.qdrant_url,
            "location": config.qdrant_location,
        },
        "pdf_inspector": describe_pdf_inspector_runtime(),
    }
    if check_store:
        store = QdrantVectorStore(
            url=config.qdrant_url,
            location=config.qdrant_location,
            api_key=config.qdrant_api_key,
            collection_name=collection_name,
            dense_dimensions=dimensions,
        )
        try:
            await store.ensure_collection()
            payload["store_health"] = await store.check_health()
        finally:
            await store.close()
    return payload


def _config_from_args(args: argparse.Namespace) -> RAGCoreConfig:
    return RAGCoreConfig(
        qdrant_url=args.qdrant_url,
        qdrant_location=args.qdrant_location,
        qdrant_api_key=args.qdrant_api_key or "",
        qdrant_collection=args.qdrant_collection,
        qdrant_dimension_aware_collection=args.dimension_aware_collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        reranker_provider=args.reranker_provider,
        reranker_model=args.reranker_model,
    )


def _emit_doctor(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    embedding = _require_mapping(payload.get("embedding"))
    reranker = _require_mapping(payload.get("reranker"))
    qdrant = _require_mapping(payload.get("qdrant"))
    print(f"Collection: {payload.get('collection_name')}")
    print(
        "Embedding: "
        f"{embedding.get('provider')} / {embedding.get('model')} / {embedding.get('dimensions')}d"
    )
    print(
        "Reranker: "
        f"requested={reranker.get('requested')} "
        f"effective={reranker.get('effective')} "
        f"reason={reranker.get('fallback_reason') or 'none'}"
    )
    print(f"Qdrant URL: {qdrant.get('url') or 'none'}")
    print(f"Qdrant Location: {qdrant.get('location') or 'none'}")
    store_health = payload.get("store_health")
    if isinstance(store_health, dict):
        print(
            "Store Health: "
            f"healthy={store_health.get('healthy')} "
            f"points={store_health.get('points_count', 0)}"
        )


def _emit_manifest(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    document = _require_mapping(payload.get("document"))
    entry = _require_mapping(payload.get("manifest_entry"))
    print(f"Document ID: {document.get('document_id')}")
    print(f"Namespace: {document.get('namespace')}")
    print(f"Corpus: {document.get('corpus_id')}")
    print(f"Document Key: {entry.get('document_key')}")
    print(f"Chunks: {entry.get('chunk_count')}")
    print(f"Parser: {entry.get('parser') or 'unknown'}")
    print(f"Needs OCR: {entry.get('needs_ocr')}")


def _env_or_none(name: str, *, default: str | None = None) -> str | None:
    value = get_env(name, default)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_or_default(name: str, default: str) -> str:
    return (get_env(name, default) or default).strip()


def _env_or_int(name: str) -> int | None:
    raw = get_env(name)
    if raw is None or not raw.strip():
        return None
    return get_env_int(name, 0) or None


def _parse_metadata_fields(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError("metadata entries must use KEY=VALUE")
        parsed[key.strip()] = raw
    return parsed


def _dataclass_payload(value: Any) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    return dict(vars(value))


def _cli_error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {exc.filename or exc}"
    return str(exc)


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
