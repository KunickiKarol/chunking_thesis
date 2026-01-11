from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoTokenizer
import yaml
from typing import List

def chunking_fixed_size(
    text: str,
    model_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """
    Dzieli tekst na chunki liczone tokenami za pomocą tokenizer'a modelu HF.
    Jeśli któryś z parametrów jest None, pobiera wartość z params.yaml.

    Args:
        text (str): tekst do podziału
        model_name (str | None): nazwa/model HF używany do tokenizacji
        chunk_size (int | None): maksymalna liczba tokenów na chunk
        chunk_overlap (int | None): overlap w tokenach między chunkami

    Returns:
        List[str]: lista chunków tekstu
    """
    import math

    # 🔹 Wczytujemy params.yaml tylko jeśli potrzebne
    if model_name is None or chunk_size is None or chunk_overlap is None:
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        chunking = params.get("chunking", {})

        if model_name is None:
            model_name = chunking.get("tokenizer_model_name", "gpt2")
        if chunk_size is None:
            chunk_size = chunking.get("chunk_size", 500)
        if chunk_overlap is None:
            chunk_overlap = chunking.get("chunk_overlap", 50)

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap musi być mniejszy niż chunk_size")

    # wczytujemy tokenizer z Hugging Face
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tworzymy splitter który liczy długość po tokenach HF
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # dzielimy tekst
    chunks = text_splitter.split_text(text)

    return chunks
