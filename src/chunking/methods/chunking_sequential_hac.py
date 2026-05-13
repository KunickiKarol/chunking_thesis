from typing import Any, Dict, List

import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.sparse import diags
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from src.chunking.methods.register import register_chunker

_model_cache: Dict[str, SentenceTransformer] = {}


@register_chunker("sequential_hac")
def chunking_sequential_hac(text: str, **params) -> List[str]:
    """
    Dzieli tekst na chunki metodą Sequential Hierarchical Agglomerative Clustering (HAC).

    Używa Single-Linkage HAC z ograniczeniem łączności do sąsiednich zdań,
    co zapewnia semantycznie spójne i sekwencyjnie ciągłe chunki.

    Params:
        text (str): Tekst do podzielenia.
        **params:
            model_name (str): ID modelu HuggingFace (domyślnie: 'BAAI/bge-m3').
            distance_threshold (float): Próg odległości linkage; niższy = więcej chunków
                                        (domyślnie: 0.3).
            batch_size (int): Rozmiar batcha podczas generowania embeddingów (domyślnie: 32).

    Returns:
        List[str]: Lista chunków tekstowych.
    """
    model_name = params.get("model_name")
    distance_threshold = float(params.get("distance_threshold"))
    batch_size = int(params.get("batch_size"))

    # Segmentacja na zdania
    sentences = sent_tokenize(text)
    n_sentences = len(sentences)

    if n_sentences == 0:
        return []
    if n_sentences == 1:
        return [sentences[0]]

    # Wczytanie modelu (cache dla wielokrotnego użycia)
    if model_name not in _model_cache:
        print(f"Ładowanie modelu: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)

    model = _model_cache[model_name]

    # Generowanie embeddingów
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Macierz połączeń — tylko sąsiednie zdania mogą być łączone
    connectivity = diags([1, 1], [-1, 1], shape=(n_sentences, n_sentences), dtype=int)

    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="single",
        connectivity=connectivity,
    )
    labels = clustering.fit_predict(embeddings)

    # Grupowanie zdań w chunki według etykiet
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_label = labels[0]

    for i, label in enumerate(labels):
        if label == current_label:
            current_sentences.append(sentences[i])
        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_label = label

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
