__all__ = [
    'CohereReranker',
    'FastEmbedSparseEmbedder',
    'NoOpReranker',
    'OpenAIEmbeddingProvider',
    'QdrantVectorStore',
    'VoyageEmbeddingProvider',
    'ZeroEntropyEmbeddingProvider',
    'ZeroEntropyReranker',
    'create_embedding_provider',
    'create_reranker',
]


def __getattr__(name: str):
    if name in {
        'OpenAIEmbeddingProvider',
        'VoyageEmbeddingProvider',
        'ZeroEntropyEmbeddingProvider',
        'create_embedding_provider',
    }:
        from .embedding import OpenAIEmbeddingProvider, create_embedding_provider
        from .voyage import VoyageEmbeddingProvider
        from .zeroentropy import ZeroEntropyEmbeddingProvider

        return {
            'OpenAIEmbeddingProvider': OpenAIEmbeddingProvider,
            'VoyageEmbeddingProvider': VoyageEmbeddingProvider,
            'ZeroEntropyEmbeddingProvider': ZeroEntropyEmbeddingProvider,
            'create_embedding_provider': create_embedding_provider,
        }[name]
    if name == 'FastEmbedSparseEmbedder':
        from .sparse import FastEmbedSparseEmbedder

        return FastEmbedSparseEmbedder
    if name == 'QdrantVectorStore':
        from .vector_store import QdrantVectorStore

        return QdrantVectorStore
    if name in {'NoOpReranker', 'CohereReranker', 'ZeroEntropyReranker', 'create_reranker'}:
        from .reranker import CohereReranker, NoOpReranker, create_reranker
        from .zeroentropy import ZeroEntropyReranker

        return {
            'NoOpReranker': NoOpReranker,
            'CohereReranker': CohereReranker,
            'ZeroEntropyReranker': ZeroEntropyReranker,
            'create_reranker': create_reranker,
        }[name]
    raise AttributeError(name)
