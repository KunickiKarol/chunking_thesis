import logging
from typing import List

from src.chunking.methods.register import register_chunker
from src.tools.chunk import Chunk, trim_bounds
from src.tools.models_cache import get_neural_chunker

logger = logging.getLogger(__name__)


@register_chunker("neural")
def chunking_neural(text: str, **params) -> List[Chunk]:
    """
    Semantic chunking using Chonkie NeuralChunker.
    """
    neural_model = params.get("neural_model")
    stride = params.get("stride")
    min_characters_per_chunk = params.get("min_characters_per_chunk")

    neural_chunker = get_neural_chunker(neural_model, stride, min_characters_per_chunk)

    try:
        chunks = neural_chunker.chunk(text)
        return [
            Chunk(*trim_bounds(chunk.text, chunk.start_index, chunk.end_index))
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]
    except Exception as e:
        logger.error(f"[ERROR] NeuralChunker failed: {e}")
        if not text.strip():
            return []
        text_stripped, start, end = trim_bounds(text, 0, len(text))
        return [Chunk(text_stripped, start, end)]
