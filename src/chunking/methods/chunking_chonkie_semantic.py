from typing import List

from src.chunking.methods.chunking_lumber import Chunk
from src.chunking.methods.register import register_chunker
from src.tools.chunk import trim_bounds
from src.tools.models_cache import get_semantic_chunker


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
    filter_window = params.get("filter_window")
    filter_polyorder = params.get("filter_polyorder")
    filter_tolerance = params.get("filter_tolerance")

    chunker = get_semantic_chunker(
        embedding_model_name=embedding_model,  # Default model
        chunk_size=chunk_size,  # Maximum tokens per chunks
        threshold=threshold,  # Similarity threshold (0-1)
        similarity_window=similarity_window,  # Window for similarity calculation
        skip_window=skip_window,  # Skip-and-merge window (0=disabled)z
        filter_window=filter_window,  # Window for smoothing the similarity curve
        filter_polyorder=filter_polyorder,  # Polynomial order for smoothing
        filter_tolerance=filter_tolerance,  # Tolerance for peak detection after smoothing
    )

    final_chunks = chunker.chunk(text)

    return [Chunk(*trim_bounds(chunk.text, chunk.start_index, chunk.end_index)) for chunk in final_chunks]
