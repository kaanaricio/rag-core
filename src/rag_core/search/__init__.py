__all__ = [
    'IndexRequest',
    'IndexResult',
    'LexicalSidecarRecord',
    'PortableLexicalSidecar',
    'QdrantIndexer',
    'SearchOrchestrator',
    'SearchRequest',
    'SearchResult',
]


def __getattr__(name: str):
    if name in {'IndexRequest', 'IndexResult', 'QdrantIndexer'}:
        from .indexer import IndexRequest, IndexResult, QdrantIndexer

        return {
            'IndexRequest': IndexRequest,
            'IndexResult': IndexResult,
            'QdrantIndexer': QdrantIndexer,
        }[name]
    if name in {'SearchOrchestrator', 'SearchRequest'}:
        from .searcher import SearchOrchestrator, SearchRequest

        return {
            'SearchOrchestrator': SearchOrchestrator,
            'SearchRequest': SearchRequest,
        }[name]
    if name in {'LexicalSidecarRecord', 'PortableLexicalSidecar'}:
        from .lexical_sidecar import LexicalSidecarRecord, PortableLexicalSidecar

        return {
            'LexicalSidecarRecord': LexicalSidecarRecord,
            'PortableLexicalSidecar': PortableLexicalSidecar,
        }[name]
    if name == 'SearchResult':
        from .types import SearchResult

        return SearchResult
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
