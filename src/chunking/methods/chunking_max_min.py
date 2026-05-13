from typing import Any, Dict, List

import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.methods.register import register_chunker


def _process_sentences(
    sentences: List[str],
    embeddings: np.ndarray,
    fixed_threshold: float,
    c: float,
    init_constant: float,
) -> List[List[str]]:
    """Grupuje zdania w paragrafy na podstawie podobieństwa semantycznego."""

    paragraphs: List[List[str]] = []
    current_paragraph = [sentences[0]]
    cluster_start, cluster_end = 0, 1
    pairwise_min = -float("inf")

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
            adjusted_threshold = pairwise_min * c * expit((cluster_end - cluster_start) - 1)
            new_sentence_similarity = np.max(new_sentence_similarities)
            pairwise_min = min(np.min(new_sentence_similarities), pairwise_min)
        else:
            adjusted_threshold = 0
            pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
            new_sentence_similarity = init_constant * pairwise_min

        if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float("inf")

    paragraphs.append(current_paragraph)
    return paragraphs


@register_chunker("max_min")
def chunking_max_min(text: str, **params) -> List[str]:
    """
    Dzieli tekst na semantyczne chunki algorytmem MaxMin.

    Parametry (przez **params):
        model_name     (str):   Model HuggingFace do embeddingów. Domyślnie "BAAI/bge-m3".
        hard_threshold (float): Stały próg podobieństwa cosinus.    Domyślnie 0.6.
        c_param        (float): Współczynnik relaksacji progu.       Domyślnie 0.9.
        init_const     (float): Stała inicjalna dla klastra=1.       Domyślnie 1.5.

    Zwraca:
        List[str]: Lista chunków (każdy chunk to string).
    """
    model_name = params.get("model_name")
    hard_threshold = float(params.get("hard_threshold"))
    c_param = float(params.get("c_param"))
    init_const = float(params.get("init_const"))
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)

    paragraphs = _process_sentences(
        sentences=sentences,
        embeddings=embeddings,
        fixed_threshold=hard_threshold,
        c=c_param,
        init_constant=init_const,
    )

    return [" ".join(paragraph) for paragraph in paragraphs]
