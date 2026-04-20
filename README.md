# rag-core

`rag-core` is a standalone ingest, indexing, and retrieval engine for serious RAG workloads.

It gives you the practical core of a document RAG stack:

- local input-to-markdown parsing
- chunk preparation with optional contextualized embedding text
- dense + sparse indexing into Qdrant
- hybrid retrieval with optional reranking
- PDF Inspector-first PDF routing with selective OCR hooks
- stable content identity and corpus manifests

It is not a hosted platform or app framework. It does not ship auth, queues, app APIs, or UI.

## What It Is

- a Python package you can embed inside your own app
- opinionated around local parsing, Qdrant, and hybrid retrieval
- practical about real documents instead of pretending every input is universally understood

## What It Is Not

- not a managed RAG service
- not a workflow/orchestration system
- not a universal document parser
- not an app shell

## Install

From a source checkout:

```bash
git clone https://github.com/kaanaricio/rag-core.git
cd rag-core
uv sync
```

With dev tools:

```bash
uv sync --group dev
```

With optional extras:

```bash
uv sync --extra semantic
uv sync --extra rerank
uv sync --extra html
uv sync --extra voyage
uv sync --extra zeroentropy
```

You can combine them:

```bash
uv sync --group dev --extra rerank --extra semantic
```

## Quickstart (First Run)

Run an end-to-end demo locally with no API keys or external Qdrant instance:

```bash
uv sync
uv run rag-core doctor --json
uv run rag-core demo
```

For PDF routing + OCR inspection on a real file from a repo checkout:

```bash
uv run python -m examples.pdf_ocr_path /path/to/local.pdf
```

The built-in demo ships inside the package. The repository also includes checkout-only example programs under `examples/`.

## Local Qdrant For App Integration

When you move past the in-memory demos, start Qdrant locally first:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

Then point `RAGCoreConfig` at it with `qdrant_url="http://localhost:6333"`.

## CLI

The package now ships a small CLI for first-run checks and file previews:

```bash
uv run rag-core doctor --json
uv run rag-core demo --json
uv run rag-core doctor --check-store --qdrant-location :memory:
uv run rag-core manifest /path/to/your.pdf --namespace acme --corpus-id help-center --json
```

`doctor` stays config-level by default, so it works without vendor API keys. Add `--check-store` when you want it to create/check the configured Qdrant collection.

## PDF Path

PDF parsing uses the local converter path.

- `PDF Inspector` is preferred by default when its CLI is available
- `PyMuPDF` is the fallback path
- mixed PDFs can keep extracted text and still flag only the pages that need OCR

Relevant env knobs already read by the package:

- `PDF_INSPECTOR_MODE`
- `PDF_INSPECTOR_BINARY_PATH`
- `PDF_INSPECTOR_TIMEOUT_MS`
- `PDF_INSPECTOR_MAX_BYTES`

## Providers And Config

Current default runtime shape:

- dense embeddings: OpenAI
- sparse retrieval: FastEmbed BM25, with optional SPLADE channel
- vector store: Qdrant
- reranker: no-op by default, optional Cohere or Voyage

Important boundary:

- `rag-core` does not auto-build `RAGCoreConfig` from general app env like `QDRANT_URL` or `OPENAI_API_KEY`
- your app should read its env and pass explicit config values into `RAGCoreConfig`
- current provider-level env fallbacks inside the package are limited to things like `COHERE_API_KEY`, sparse model names, and PDF Inspector settings

Useful env knobs:

- `COHERE_API_KEY`
- `RERANKER_STRICT_PROVIDER`
- `SPARSE_EMBEDDING_MODEL`
- `SPARSE_EMBEDDING_MODEL_BM25`
- `SPARSE_EMBEDDING_MODEL_SPLADE`

## Minimal Integration

This example assumes:

- Qdrant is running on `http://localhost:6333`
- `OPENAI_API_KEY` is already set in your environment
- you are indexing a real local file you own

```python
import asyncio

from rag_core import RAGCore, RAGCoreConfig


async def main() -> None:
    core = RAGCore(
        RAGCoreConfig(
            qdrant_url="http://localhost:6333",
            qdrant_collection="product_docs",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            reranker_provider="none",
            contextualize=True,
        )
    )

    try:
        await core.ensure_ready()

        await core.ingest_file(
            "/path/to/faq.pdf",
            namespace="acme",
            corpus_id="help-center",
        )

        hits = await core.search(
            query="How does billing work?",
            namespace="acme",
            corpus_ids=["help-center"],
            limit=5,
        )

        for hit in hits:
            print(hit.score, hit.title or hit.document_id, hit.text[:120])
    finally:
        await core.close()


asyncio.run(main())
```

## OCR Helpers

`rag-core` exposes `CommandOcrProvider` plus helper builders for Mistral and Gemini OCR command wrappers. Use them when you want OCR without hard-wiring a specific SDK into the package core.

```python
from rag_core.documents import build_gemini_ocr_provider, build_mistral_ocr_provider

mistral_ocr = build_mistral_ocr_provider()
gemini_ocr = build_gemini_ocr_provider()
```

## Customizing Providers

Start with config changes first:

- change `embedding_provider`, `embedding_model`, and `embedding_dimensions`
- switch between local and remote Qdrant with `qdrant_location` or `qdrant_url`
- enable reranking with `reranker_provider="cohere"` or `reranker_provider="voyage"`

If your app needs different provider wiring, inject your own implementations into `RAGCore(...)`:

- `embedding_provider`
- `sparse_embedder`
- `vector_store`
- `reranker`
- `ocr_provider`

## Examples

If you are working from a source checkout, the repository also includes:

- [`examples/minimal_app.py`](examples/minimal_app.py): smallest runnable ingest + search app. Runnable via `python -m examples.minimal_app`.
- [`examples/corpus_lifecycle.py`](examples/corpus_lifecycle.py): ingest, manifest, search, and delete flow around namespace/corpus partitioning and stable document keys. Runnable via `python -m examples.corpus_lifecycle`.
- [`examples/pdf_ocr_path.py`](examples/pdf_ocr_path.py): PDF routing and OCR setup path. Runnable via `python -m examples.pdf_ocr_path /path/to/local.pdf`.

## Troubleshooting

- `rag-core doctor --check-store` fails immediately:
  - check that you set exactly one of `--qdrant-url` or `--qdrant-location`
  - for the local service path, confirm Qdrant is running on `http://localhost:6333`
- `rag-core manifest ...` or PDF examples report `pdf-inspector` unavailable:
  - this is expected when the CLI is not installed
  - `rag-core` falls back to the local PyMuPDF path
- OpenAI, Cohere, Voyage, or ZeroEntropy providers fail at runtime:
  - install the matching optional extra first
  - then pass explicit config or set the provider-specific API key env var
- `mypy` or `ruff` starts reading build artifacts:
  - remove local `build/` and `dist/` directories if you created them manually
  - current repo config excludes them from normal checks

## Validation

Run the existing lightweight checks:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src tests examples
```
