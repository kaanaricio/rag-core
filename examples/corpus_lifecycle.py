from __future__ import annotations

import asyncio
from dataclasses import asdict

from examples import build_demo_core
from rag_core import CorpusManifestEntry, RAGCore, SearchResult


def manifest_key(*, namespace: str, corpus_id: str, document_key: str) -> str:
    return f"{namespace}:{corpus_id}:{document_key}"


async def ingest_into_manifest(
    core: RAGCore,
    *,
    manifest: dict[str, CorpusManifestEntry],
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    namespace: str,
    corpus_id: str,
    metadata: dict[str, str] | None = None,
) -> CorpusManifestEntry:
    ingested = await core.ingest_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        namespace=namespace,
        corpus_id=corpus_id,
        metadata=metadata,
    )
    document_key = ingested.document_key or ingested.filename
    key = manifest_key(
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=document_key,
    )
    entry = core.build_manifest_entry(document=ingested)
    manifest[key] = entry
    return entry


async def search_corpus(
    core: RAGCore,
    *,
    entry: CorpusManifestEntry,
    query: str,
    limit: int = 5,
    rerank: bool = False,
) -> list[SearchResult]:
    return await core.search(
        query=query,
        namespace=entry.namespace,
        corpus_ids=[entry.corpus_id],
        limit=limit,
        rerank=rerank,
    )


async def delete_from_manifest(
    core: RAGCore,
    *,
    manifest: dict[str, CorpusManifestEntry],
    key: str,
) -> CorpusManifestEntry:
    entry = manifest.pop(key)
    await core.delete_document(
        document_id=entry.document_id,
        namespace=entry.namespace,
        corpus_id=entry.corpus_id,
    )
    return entry


def manifest_row(entry: CorpusManifestEntry) -> dict[str, object]:
    return asdict(entry)


async def run_demo() -> None:
    """Run a local end-to-end lifecycle demo without external services."""
    core = build_demo_core(collection="corpus_lifecycle")
    manifest: dict[str, CorpusManifestEntry] = {}

    try:
        await core.ensure_ready()
        billing_entry = await ingest_into_manifest(
            core,
            manifest=manifest,
            file_bytes=b"Billing is due monthly. Payment methods include card and ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
            metadata={"source": "quickstart"},
        )
        await ingest_into_manifest(
            core,
            manifest=manifest,
            file_bytes=b"Shipping times are 3-5 business days in the continental US.",
            filename="shipping.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
            metadata={"source": "quickstart"},
        )

        hits = await search_corpus(
            core,
            entry=billing_entry,
            query="How can I pay invoices?",
            limit=3,
        )
        print("Manifest keys:")
        for key in sorted(manifest):
            print(f"- {key}")
        print("\nTop hits:")
        for hit in hits:
            title = hit.title or hit.document_id or "unknown"
            print(f"- {hit.score:.3f} {title}: {hit.text[:80]}")

        first_key = sorted(manifest)[0]
        deleted = await delete_from_manifest(core, manifest=manifest, key=first_key)
        print(f"\nDeleted document: {deleted.document_id}")
        print(f"Remaining manifest entries: {len(manifest)}")
    finally:
        await core.close()


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
