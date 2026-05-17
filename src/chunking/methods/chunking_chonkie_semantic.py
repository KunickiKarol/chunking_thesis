from typing import List

from chonkie import NeuralChunker, SemanticChunker, TokenChunker
from transformers import AutoTokenizer

from src.chunking.methods.chunking_lumber import Chunk
from src.chunking.methods.register import register_chunker
from src.tools.chunk import trim_bounds


@register_chunker("chonkie_semantic")
def chunking_neural(
    text: str,
    **params,
) -> List[str]:
    """
    Hybrid chunking:
    1. TokenChunker (fixed-size token splitting)
    2. NeuralChunker (semantic refinement)
    """

    embedding_model = params.get("embedding_model")
    threshold = params.get("threshold")
    chunk_size = params.get("chunk_size")
    similarity_window = params.get("similarity_window")
    skip_window = params.get("skip_window")

    chunker = SemanticChunker(
        embedding_model=embedding_model,  # Default model
        threshold=threshold,  # Similarity threshold (0-1)
        chunk_size=chunk_size,  # Maximum tokens per chunk
        similarity_window=similarity_window,  # Window for similarity calculation
        skip_window=skip_window,  # Skip-and-merge window (0=disabled)
    )

    final_chunks = chunker.chunk(text)

    return [
        Chunk(*trim_bounds(chunk.text, chunk.start_index, chunk.end_index))
        for chunk in final_chunks
    ]
