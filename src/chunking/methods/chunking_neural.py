from typing import List

from chonkie import NeuralChunker
from src.chunking.methods.register import register_chunker
from src.tools.chunk import Chunk, trim_bounds


@register_chunker("neural")
def chunking_neural(
    text: str,
    **params,
) -> List[Chunk]:
    """
    Semantic chunking using Chonkie NeuralChunker.

    Uses:
    - stride
    - min_characters_per_chunk

    No manual pre-splitting or token windowing.
    """

    neural_model = params.get("neural_model")
    stride = params.get("stride")
    min_characters_per_chunk = params.get("min_characters_per_chunk")

    # =========================================================
    # NeuralChunker
    # =========================================================
    neural_chunker = NeuralChunker(
        model=neural_model,
        stride=stride,
        min_characters_per_chunk=min_characters_per_chunk,
    )

    try:
        chunks = neural_chunker.chunk(text)

        return [
            Chunk(*trim_bounds(chunk.text, chunk.start_index, chunk.end_index))
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]

    except Exception as e:
        print(f"[ERROR] NeuralChunker failed: {e}")

        # fallback
        if not text.strip():
            return []

        text_stripped, start, end = trim_bounds(text, 0, len(text))

        return [Chunk(text_stripped, start, end)]