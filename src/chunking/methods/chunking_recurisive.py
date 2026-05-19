from typing import List

from chonkie import RecursiveChunker, RecursiveRules

from src.chunking.methods.chunking_lumber import Chunk
from src.chunking.methods.register import register_chunker
from src.tools.chunk import trim_bounds

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="chonkie.*"
)

@register_chunker("recursive")
def chunking_recursive(
    text: str,
    *,
    chunk_size: int,
    min_characters_per_chunk: int,
) -> List[str]:
    """
    Dzieli tekst metodą recursive chunking.

    Args:
        text (str): tekst wejściowy
        chunk_size (int): maksymalny rozmiar chunka
        min_characters_per_chunk (int): minimalna liczba znaków per chunk

    Returns:
        List[str]: lista chunków
    """

    if min_characters_per_chunk >= chunk_size:
        raise ValueError("min_characters_per_chunk must be smaller than chunk_size")

    chunker = RecursiveChunker(tokenizer="character",
        chunk_size=chunk_size,
        rules=RecursiveRules(),
        min_characters_per_chunk=min_characters_per_chunk,
    )
    return [
        Chunk(*trim_bounds(chunk.text, chunk.start_index, chunk.end_index))
        for chunk in chunker.chunk(text)
    ]
