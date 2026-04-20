from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.core_file_io import read_file_bytes
from rag_core.core_lifecycle import (
    compute_content_sha256,
    resolve_document_id,
    resolve_document_key,
    resolve_ingest_state,
)
from rag_core.core_manifest import (
    build_corpus_manifest,
    build_manifest_entry,
    build_preview_document,
)
from rag_core.core_models import (
    CorpusManifest,
    CorpusManifestEntry,
    IngestedDocument,
    OcrRoutingSignal,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
    RAGCoreConfig,
)
from rag_core.core_prepare import parse_document_bytes, prepare_document_bytes
from rag_core.core_runtime import build_runtime_description, resolve_collection_name
from rag_core.core_sidecar import build_sidecar_records
from rag_core.search.indexer import IndexRequest, QdrantIndexer
from rag_core.search.lexical_sidecar import PortableLexicalSidecar
from rag_core.search.searcher import SearchOrchestrator, SearchRequest
from rag_core.search.types import SearchResult

if TYPE_CHECKING:
    from rag_core.documents.ocr import OcrProvider
    from rag_core.search.types import (
        EmbeddingProvider,
        RerankerProvider,
        SearchSidecar,
        SparseEmbedder,
        VectorStore,
    )


class RAGCore:
    def __init__(
        self,
        config: RAGCoreConfig,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: RerankerProvider | None = None,
        ocr_provider: OcrProvider | None = None,
        search_sidecar: SearchSidecar | None = None,
    ) -> None:
        from rag_core.search.providers import (
            FastEmbedSparseEmbedder,
            QdrantVectorStore,
            create_embedding_provider,
            create_reranker,
        )

        self._config = config
        self._embedding = embedding_provider or create_embedding_provider(
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
        )
        self._sparse = sparse_embedder or FastEmbedSparseEmbedder()
        self._ocr = ocr_provider
        self._collection_name = resolve_collection_name(
            base_name=config.qdrant_collection,
            model_name=self._embedding.model_name,
            dimensions=self._embedding.dimensions,
            dimension_aware=config.qdrant_dimension_aware_collection,
        )
        self._store = vector_store or QdrantVectorStore(
            url=config.qdrant_url,
            location=config.qdrant_location,
            api_key=config.qdrant_api_key,
            collection_name=self._collection_name,
            dense_dimensions=self._embedding.dimensions,
        )
        self._reranker = reranker or create_reranker(
            provider=config.reranker_provider,
            model=config.reranker_model,
            api_key=config.reranker_api_key,
        )
        self._sidecar = search_sidecar
        if self._sidecar is None and config.enable_exact_match_sidecar:
            self._sidecar = PortableLexicalSidecar([])
        self._indexer = QdrantIndexer(
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            vector_store=self._store,
        )
        self._search = SearchOrchestrator(
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            vector_store=self._store,
            reranker=self._reranker,
            sidecar=self._sidecar,
        )

    async def ensure_ready(self) -> None:
        await self._store.ensure_collection()

    async def check_health(self) -> dict[str, object]:
        return await self._store.check_health()

    async def close(self) -> None:
        await self._store.close()

    async def parse_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> ParsedDocument:
        return await parse_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
        )

    async def prepare_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> PreparedDocument:
        return await prepare_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            contextualize=self._config.contextualize,
            ocr_provider=self._ocr,
        )

    async def prepare_file(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
    ) -> PreparedDocument:
        file_path = Path(path)
        detected_mime_type = mime_type
        if detected_mime_type is None:
            detected_mime_type, _ = mimetypes.guess_type(str(file_path))
        return await self.prepare_bytes(
            file_bytes=await read_file_bytes(file_path),
            filename=file_path.name,
            mime_type=detected_mime_type or "application/octet-stream",
            path=str(file_path),
        )

    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestedDocument:
        resolved_document_key = resolve_document_key(
            filename=filename,
            path=path,
            document_key=document_key,
        )
        resolved_document_id = resolve_document_id(
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=resolved_document_key,
            document_id=document_id,
        )
        content_sha256 = compute_content_sha256(file_bytes)
        existing = await self._store.get_document_record(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=resolved_document_id,
        )
        ingest_state, should_index = resolve_ingest_state(
            existing,
            content_sha256=content_sha256,
        )
        if not should_index:
            prepared = await self.prepare_bytes(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=mime_type,
                path=path,
            )
            return IngestedDocument(
                document_id=resolved_document_id,
                corpus_id=corpus_id,
                namespace=namespace,
                chunk_count=existing.chunk_count if existing is not None else 0,
                filename=filename,
                mime_type=mime_type,
                document_key=resolved_document_key,
                content_sha256=content_sha256,
                ingest_state=ingest_state,
                replaced_existing=False,
                collection_name=self._collection_name,
                embedding_model=self._embedding.model_name,
                ocr=prepared.ocr,
                metadata=_merge_document_metadata(prepared.metadata, metadata),
            )

        prepared = await self.prepare_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
        )
        result = await self._indexer.index_document(
            IndexRequest(
                document_id=resolved_document_id,
                document_key=resolved_document_key,
                content_sha256=content_sha256,
                existing_chunk_count=existing.chunk_count if existing is not None else None,
                corpus_id=corpus_id,
                namespace=namespace,
                text=prepared.markdown,
                filename=prepared.filename,
                mime_type=prepared.mime_type,
                source_type=self._config.source_type,
                path=prepared.path,
                document_path=prepared.path,
                extra_fields=dict(metadata or {}) or None,
                embedding_model=self._embedding.model_name,
                pre_chunked_texts=[chunk.text for chunk in prepared.chunks],
                embedding_chunk_texts=[chunk.embedding_text for chunk in prepared.chunks],
            )
        )
        if self._sidecar is not None:
            self._sidecar.delete_document(
                namespace=namespace,
                document_id=resolved_document_id,
                corpus_id=corpus_id,
            )
            self._sidecar.upsert_records(
                build_sidecar_records(
                    namespace=namespace,
                    point_ids=result.point_ids,
                    point_payloads=result.point_payloads,
                )
            )
        return IngestedDocument(
            document_id=resolved_document_id,
            corpus_id=corpus_id,
            namespace=namespace,
            chunk_count=result.chunk_count,
            filename=prepared.filename,
            mime_type=prepared.mime_type,
            document_key=result.document_key,
            content_sha256=result.content_sha256,
            ingest_state=ingest_state,
            replaced_existing=existing is not None,
            collection_name=self._collection_name,
            embedding_model=self._embedding.model_name,
            ocr=prepared.ocr,
            metadata=_merge_document_metadata(prepared.metadata, metadata),
        )

    async def ingest_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestedDocument:
        file_path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return await self.ingest_bytes(
            file_bytes=await read_file_bytes(file_path),
            filename=file_path.name,
            mime_type=mime_type or "application/octet-stream",
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            path=str(file_path),
            metadata=metadata,
        )

    def build_manifest_entry(
        self,
        *,
        document: IngestedDocument,
    ) -> CorpusManifestEntry:
        return build_manifest_entry(document)

    async def manifest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CorpusManifestEntry:
        prepared = await self.prepare_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
        )
        preview = build_preview_document(
            file_bytes=file_bytes,
            prepared=prepared,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
            collection_name=self._collection_name,
            embedding_model=self._embedding.model_name,
        )
        return build_manifest_entry(preview)

    async def manifest_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CorpusManifestEntry:
        file_path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return await self.manifest_bytes(
            file_bytes=await read_file_bytes(file_path),
            filename=file_path.name,
            mime_type=mime_type or "application/octet-stream",
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            path=str(file_path),
            metadata=metadata,
        )

    def build_corpus_manifest(
        self,
        *,
        namespace: str,
        corpus_id: str,
        documents: list[IngestedDocument],
    ) -> CorpusManifest:
        return build_corpus_manifest(
            namespace=namespace,
            corpus_id=corpus_id,
            collection_name=self._collection_name,
            embedding_provider=self._config.embedding_provider,
            embedding_model=self._embedding.model_name,
            embedding_dimensions=self._embedding.dimensions,
            documents=documents,
        )

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int = 10,
        document_ids: list[str] | None = None,
        rerank: bool = True,
        use_sidecar: bool = True,
    ) -> list[SearchResult]:
        return await self._search.search(
            SearchRequest(
                query=query,
                corpus_ids=corpus_ids,
                namespace=namespace,
                limit=limit,
                document_ids=document_ids,
                rerank=rerank,
                use_sidecar=use_sidecar,
            )
        )

    async def delete_document(
        self,
        *,
        document_id: str,
        namespace: str,
        corpus_id: str,
    ) -> None:
        await self._indexer.delete_document(
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
        )
        if self._sidecar is not None:
            self._sidecar.delete_document(
                namespace=namespace,
                document_id=document_id,
                corpus_id=corpus_id,
            )

    def describe_runtime(self) -> dict[str, object]:
        return build_runtime_description(
            collection_name=self._collection_name,
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            reranker=self._reranker,
            ocr_provider=self._ocr,
        )


def _merge_document_metadata(
    prepared_metadata: dict[str, object],
    metadata: dict[str, str] | None,
) -> dict[str, object]:
    return {**prepared_metadata, **dict(metadata or {})}


__all__ = [
    "CorpusManifest",
    "CorpusManifestEntry",
    "IngestedDocument",
    "OcrRoutingSignal",
    "ParsedDocument",
    "PreparedChunk",
    "PreparedDocument",
    "RAGCore",
    "RAGCoreConfig",
    "SearchResult",
]
