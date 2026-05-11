from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.methods.register import register_chunker


@register_chunker("recursive")
def chunking_recursive(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Dzieli tekst metodą recursive chunking.

    Args:
        text (str): tekst wejściowy
        chunk_size (int): maksymalny rozmiar chunka
        overlap (int): overlap między chunkami

    Returns:
        List[str]: lista chunków
    """

    if overlap >= chunk_size:
        raise ValueError("overlap musi być mniejszy niż chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # zero-width space
            "\uff0c",  # fullwidth comma
            "\u3001",  # ideographic comma
            "\uff0e",  # fullwidth full stop
            "\u3002",  # ideographic full stop
            "",
        ],
    )

    return splitter.split_text(text)
