from dataclasses import dataclass
from typing import List

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.methods.register import register_chunker
from src.tools.chunk import Chunk, trim_bounds


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


def _tokenize_with_spans(text: str) -> List[SentenceSpan]:
    """
    Tokenizuje tekst i zachowuje oryginalne indeksy zdań.
    """
    tokenizer = PunktSentenceTokenizer()

    spans = tokenizer.span_tokenize(text)

    return [
        SentenceSpan(
            text=text[start:end],
            start=start,
            end=end,
        )
        for start, end in spans
    ]


def _process_sentences(
    sentences: List[SentenceSpan],
    embeddings: np.ndarray,
    fixed_threshold: float,
    c: float,
    init_constant: float,
) -> List[List[SentenceSpan]]:
    """
    Grupuje zdania w paragrafy na podstawie podobieństwa semantycznego.
    """
    paragraphs: List[List[SentenceSpan]] = []

    current_paragraph = [sentences[0]]

    cluster_start, cluster_end = 0, 1
    pairwise_min = -float("inf")

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(
                embeddings[i].reshape(1, -1),
                cluster_embeddings,
            )[0]

            adjusted_threshold = (
                pairwise_min
                * c
                * expit((cluster_end - cluster_start) - 1)
            )

            new_sentence_similarity = np.max(new_sentence_similarities)

            pairwise_min = min(
                np.min(new_sentence_similarities),
                pairwise_min,
            )

        else:
            adjusted_threshold = 0

            pairwise_min = cosine_similarity(
                embeddings[i].reshape(1, -1),
                cluster_embeddings,
            )[0][0]

            new_sentence_similarity = init_constant * pairwise_min

        if new_sentence_similarity > max(
            adjusted_threshold,
            fixed_threshold,
        ):
            current_paragraph.append(sentences[i])
            cluster_end += 1

        else:
            paragraphs.append(current_paragraph)

            current_paragraph = [sentences[i]]

            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float("inf")

    paragraphs.append(current_paragraph)

    return paragraphs


def _build_chunks(
    text: str,
    paragraphs: List[List[SentenceSpan]],
) -> List[Chunk]:
    """
    Buduje chunki bez utraty whitespace/newline.
    """
    chunks: List[Chunk] = []

    for paragraph in paragraphs:
        start_index = paragraph[0].start
        end_index = paragraph[-1].end

        chunk_text = text[start_index:end_index]

        chunk_text, start_index, end_index = trim_bounds(
            chunk_text,
            start_index,
            end_index,
        )

        chunks.append(
            Chunk(
                text=chunk_text,
                start_index=start_index,
                end_index=end_index,
            )
        )

    return chunks


@register_chunker("max_min")
def chunking_max_min(text: str, **params) -> List[Chunk]:
    """
    Dzieli tekst na semantyczne chunki algorytmem MaxMin.
    """

    model_name = params.get("model_name")
    hard_threshold = float(params.get("hard_threshold"))
    c_param = float(params.get("c_param"))
    init_const = float(params.get("init_const"))

    sentences = _tokenize_with_spans(text)

    if not sentences:
        return []

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        [s.text for s in sentences]
    )

    paragraphs = _process_sentences(
        sentences=sentences,
        embeddings=embeddings,
        fixed_threshold=hard_threshold,
        c=c_param,
        init_constant=init_const,
    )

    return _build_chunks(text, paragraphs)