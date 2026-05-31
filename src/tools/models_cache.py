from chonkie import NeuralChunker, SemanticChunker, SentenceTransformerEmbeddings
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer
from wrapt import lru_cache

from src.tools.tokenizer_service import TokenizerService


@lru_cache(maxsize=1)
def get_neural_chunker(
    neural_model: str,
    stride: int,
    min_characters_per_chunk: int,
) -> NeuralChunker:
    return NeuralChunker(
        model=neural_model,
        stride=stride,
        min_characters_per_chunk=min_characters_per_chunk,
    )


@lru_cache(maxsize=2)
def get_tokenizer_service(
    backend: str = None,
    model_name: str = None,
) -> TokenizerService:
    return TokenizerService(
        backend=backend,
        model_name=model_name,
    )


@lru_cache(maxsize=1)
def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    Cache SentenceTransformer models by:
    - model_name
    - device_name
    """
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_sentence_transformer_embeddings(model_name: str) -> SentenceTransformerEmbeddings:
    """
    Cache SentenceTransformer models by:
    - model_name
    - device_name
    """
    return SentenceTransformerEmbeddings(model_name)


@lru_cache(maxsize=1)
def get_semantic_chunker(
    embedding_model_name: str,
    chunk_size: int,
    threshold: float,
    similarity_window: int,
    skip_window: int,
    filter_window: int,
    filter_polyorder: int,
    filter_tolerance: float,
) -> SemanticChunker:
    """
    Cache SemanticChunker instances by full configuration.
    """

    model = get_sentence_transformer_embeddings(embedding_model_name)

    return SemanticChunker(
        embedding_model=model,
        chunk_size=chunk_size,
        threshold=threshold,
        similarity_window=similarity_window,
        skip_window=skip_window,
        filter_window=filter_window,
        filter_polyorder=filter_polyorder,
        filter_tolerance=filter_tolerance,
    )


def clear_all_caches() -> None:
    """
    Czyści cache wszystkich funkcji opartych o lru_cache w tym module.
    """
    get_neural_chunker.cache_clear()
    get_sentence_transformer.cache_clear()
    get_semantic_chunker.cache_clear()


@lru_cache(maxsize=1)
def get_reranker(model_name: str):
    """
    Cache singletonów dla rerankerów.
    Model ładowany tylko raz per model_name.
    """
    return FlagReranker(model_name, use_fp16=True)
