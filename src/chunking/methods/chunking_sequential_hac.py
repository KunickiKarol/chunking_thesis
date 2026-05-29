import logging
from typing import List

import torch
from nltk.tokenize import sent_tokenize
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering

from src.chunking.methods.register import register_chunker
from src.tools.chunk import Chunk, trim_bounds
from src.tools.models_cache import get_sentence_transformer

logger = logging.getLogger(__name__)


@register_chunker("sequential_hac")
def chunking_sequential_hac(text: str, **params) -> List[Chunk]:
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
        List[Chunk]: Lista chunków z tekstem i indeksami w oryginalnym tekście.
    """
    model_name = params.get("model_name")
    distance_threshold = float(params.get("distance_threshold"))
    batch_size = int(params.get("batch_size"))

    # Segmentacja na zdania
    sentences = sent_tokenize(text)
    n_sentences = len(sentences)

    if n_sentences == 0:
        return []

    # Wyznacz pozycje każdego zdania w oryginalnym tekście
    sentence_spans: List[tuple[int, int]] = []
    search_start = 0

    for sentence in sentences:
        start = text.find(sentence, search_start)
        end = start + len(sentence)
        sentence_spans.append((start, end))
        search_start = end

    if n_sentences == 1:
        start, end = sentence_spans[0]
        return [
            Chunk(
                text=sentences[0],
                start_index=start,
                end_index=end - 1,
            )
        ]

    # Wczytanie modelu z cache
    model = get_sentence_transformer(model_name)

    while batch_size >= 1:
        try:
            embeddings = model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            break

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            torch.cuda.empty_cache()

            if batch_size == 1:
                raise

            batch_size //= 2

            logger.info(f"CUDA OOM -> retrying with batch_size={batch_size}")
    else:
        raise RuntimeError("Failed to encode embeddings")

    # Macierz połączeń — tylko sąsiednie zdania mogą być łączone
    connectivity = diags(
        [1, 1],
        [-1, 1],
        shape=(n_sentences, n_sentences),
        dtype=int,
    )

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
    chunks: List[Chunk] = []

    current_sentences: List[str] = []
    current_spans: List[tuple[int, int]] = []
    current_label = labels[0]

    for i, label in enumerate(labels):
        if label == current_label:
            current_sentences.append(sentences[i])
            current_spans.append(sentence_spans[i])
        else:
            chunk_text = text[current_spans[0][0] : current_spans[-1][1]]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=current_spans[0][0],
                    end_index=current_spans[-1][1],
                )
            )

            current_sentences = [sentences[i]]
            current_spans = [sentence_spans[i]]
            current_label = label

    if current_sentences:
        chunk_text = text[current_spans[0][0] : current_spans[-1][1]]

        chunks.append(
            Chunk(
                text=chunk_text,
                start_index=current_spans[0][0],
                end_index=current_spans[-1][1],
            )
        )

    return [Chunk(*trim_bounds(chunk.text, chunk.start_index, chunk.end_index)) for chunk in chunks]
