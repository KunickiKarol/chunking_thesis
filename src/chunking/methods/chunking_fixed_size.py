from typing import List

from langchain_text_splitters import CharacterTextSplitter


def chunking_fixed_size(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Dzieli tekst na chunki o stałej długości (fixed size).

    Args:
        text (str): tekst wejściowy
        chunk_size (int): rozmiar chunka
        overlap (int): overlap między chunkami

    Returns:
        List[str]: lista chunków
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,  # jawnie: znaki, nie tokeny
    )

    return splitter.split_text(text)
