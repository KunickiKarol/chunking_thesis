import re
from typing import List, Optional

import nltk
import numpy as np
import torch
from chonkie import SemanticChunker
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from src.chunking.methods.chunking_lumber import Chunk
from src.chunking.methods.register import register_chunker


@register_chunker("recursive_semantic")
def chunking_semantic_recursive(
    text: str,
    **params,
) -> List[Chunk]:
    """
    Recursive Semantic Chunking using chonkie library.

    Returns:
        List[Chunk] -> final chunks with text, start_index, end_index
    """
    max_chunk_size = params.get("max_chunk_size")
    recursive_threshold = params.get("recursive_threshold")
    final_threshold = params.get("final_threshold")
    merge_threshold = params.get("merge_threshold")
    delta = params.get("delta")
    embedding_model = params.get("embedding_model")
    breakpoint_threshold_type = params.get("breakpoint_threshold_type")
    initial_breakpoint_threshold = params.get("initial_breakpoint_threshold")
    embeddings = params.get("embeddings")

    if not text or not text.strip():
        return []

    if delta <= 0:
        raise ValueError("delta must be positive")

    if merge_threshold >= recursive_threshold:
        raise ValueError("merge_threshold must be smaller than recursive_threshold")

    if final_threshold > max_chunk_size:
        raise ValueError("final_threshold must not exceed max_chunk_size")

    nltk.download("punkt_tab", quiet=True)

    # ------------------------------------------------------------------ #
    # Embeddings — chonkie accepts a model name string directly
    # ------------------------------------------------------------------ #

    # chonkie's SemanticChunker takes a model name (str) or a callable embedder.
    # We keep the embedding_model name for chonkie and also build an embedding
    # function for the merge step (numpy-based similarity).

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If the caller passes a pre-built HuggingFaceEmbeddings instance we wrap it
    # so it stays compatible with the merge step.  Otherwise we rely on chonkie's
    # own model loading and build a thin wrapper for the merge helper.

    if embeddings is not None:
        # Assume it exposes .embed_documents(List[str]) -> List[List[float]]
        def _embed(texts: List[str]) -> np.ndarray:
            return np.array(embeddings.embed_documents(texts))
    else:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(embedding_model, device=device)

        def _embed(texts: List[str]) -> np.ndarray:
            return _st_model.encode(texts, convert_to_numpy=True)

    # ------------------------------------------------------------------ #
    # chonkie SemanticChunker factory (cached by threshold)
    # ------------------------------------------------------------------ #

    _chunker_cache: dict = {}

    def _get_chunker(threshold: float) -> SemanticChunker:
        key = round(threshold, 6)
        if key not in _chunker_cache:
            _chunker_cache[key] = SemanticChunker(
                embedding_model=embedding_model if embeddings is None else embeddings,
                chunk_size=max_chunk_size,
                threshold=threshold,
                # chonkie uses "percentile" / "standard_deviation" / "interquartile"
                # map from the original breakpoint_threshold_type when provided
                threshold_type=breakpoint_threshold_type or "percentile",
                filter_window=0,  # we want to consider all pairs, not just local neighbors
                filter_polyorder=0,  # no smoothing
                filter_tolerance=0,  # no smoothing
            )
        return _chunker_cache[key]

    # ------------------------------------------------------------------ #
    # Sentence tokenizer (unchanged)
    # ------------------------------------------------------------------ #

    def tokenize_sentences(text_: str) -> List[str]:
        try:
            try:
                return nltk.sent_tokenize(text_)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                return nltk.sent_tokenize(text_)
        except ImportError:
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_) if s.strip()]

    # ------------------------------------------------------------------ #
    # Helper: find the position of a chunk text inside the *original* text,
    # starting the search from `search_from` to avoid false matches.
    # ------------------------------------------------------------------ #

    def _locate(chunk_text: str, search_from: int = 0) -> tuple[int, int]:
        """Return (start, end) of chunk_text inside `text`, starting from search_from."""
        idx = text.find(chunk_text, search_from)
        if idx == -1:
            # Fallback: search from the beginning (handles edge cases with
            # whitespace normalisation done by the splitter)
            idx = text.find(chunk_text)
        if idx == -1:
            # Last resort — return a sentinel so callers can detect failure
            return (-1, -1)
        return idx, idx + len(chunk_text)

    # ------------------------------------------------------------------ #
    # Step 1 — Segment file (returns list of (text, start, end))
    # ------------------------------------------------------------------ #

    def segment_file(text_: str) -> List[tuple[str, int, int]]:
        if len(text_) <= max_chunk_size:
            return [(text_, 0, len(text_))]

        sentences = tokenize_sentences(text_)

        segments: List[tuple[str, int, int]] = []
        current_parts: List[str] = []
        current_len = 0

        for sentence in sentences:
            slen = len(sentence)

            if slen > max_chunk_size:
                if current_parts:
                    seg_text = " ".join(current_parts)
                    start, end = _locate(seg_text)
                    segments.append((seg_text, start, end))
                    current_parts = []
                    current_len = 0

                for chunk_start in range(0, slen, max_chunk_size):
                    piece = sentence[chunk_start: chunk_start + max_chunk_size]
                    start, end = _locate(piece)
                    segments.append((piece, start, end))

                continue

            if current_len + slen > max_chunk_size:
                seg_text = " ".join(current_parts)
                start, end = _locate(seg_text)
                segments.append((seg_text, start, end))
                current_parts = []
                current_len = 0

            current_parts.append(sentence)
            current_len += slen + 1

        if current_parts:
            seg_text = " ".join(current_parts)
            start, end = _locate(seg_text)
            segments.append((seg_text, start, end))

        return segments

    # ------------------------------------------------------------------ #
    # Step 2 — Semantic split via chonkie
    # Returns list of (text, start_in_original, end_in_original)
    # ------------------------------------------------------------------ #

    def semantic_split(
        text_: str,
        threshold: float,
        search_from: int = 0,
    ) -> List[tuple[str, int, int]]:
        chunker = _get_chunker(threshold)
        chunks = chunker.chunk(text_)  # returns List[chonkie.Chunk]

        result: List[tuple[str, int, int]] = []
        cursor = search_from

        for c in chunks:
            chunk_text = c.text
            if not chunk_text.strip():
                continue

            # chonkie provides start_index / end_index relative to the input text_.
            # We shift them by search_from to get positions in the *original* text.
            if hasattr(c, "start_index") and c.start_index is not None:
                abs_start = search_from + c.start_index
                abs_end = search_from + c.end_index
            else:
                # Fallback: locate in the original text starting from cursor
                abs_start, abs_end = _locate(chunk_text, cursor)

            result.append((chunk_text, abs_start, abs_end))
            cursor = abs_end

        return result

    # ------------------------------------------------------------------ #
    # Step 3 — Recursive semantic split
    # ------------------------------------------------------------------ #

    def recursive_semantic_split(
        chunk_text: str,
        chunk_start: int,
        threshold: float,
    ) -> List[tuple[str, int, int]]:

        if len(chunk_text) <= threshold:
            return [(chunk_text, chunk_start, chunk_start + len(chunk_text))]

        next_threshold = max(threshold - delta, 1.0)

        sub_chunks = semantic_split(
            chunk_text,
            initial_breakpoint_threshold,
            search_from=chunk_start,
        )

        # No progress fallback
        if len(sub_chunks) == 1 and len(sub_chunks[0][0]) >= len(chunk_text):
            return [(chunk_text, chunk_start, chunk_start + len(chunk_text))]

        result: List[tuple[str, int, int]] = []

        for sub_text, sub_start, sub_end in sub_chunks:
            result.extend(
                recursive_semantic_split(sub_text, sub_start, next_threshold)
            )

        return result

    # ------------------------------------------------------------------ #
    # Step 4 — Merge small chunks (preserves indices)
    # ------------------------------------------------------------------ #

    def merge_small_chunks(
        chunks: List[tuple[str, int, int]]
    ) -> List[tuple[str, int, int]]:
        if len(chunks) <= 1:
            return chunks

        merged = list(chunks)  # list of (text, start, end)

        texts = [t for t, _, _ in merged]
        embeddings_cache = _embed(texts)  # (N, dim)

        i = 0
        while i < len(merged):
            text_i, start_i, end_i = merged[i]

            if len(text_i) <= merge_threshold and len(merged) > 1:
                has_prev = i > 0
                has_next = i < len(merged) - 1

                def _merge_with_prev():
                    nonlocal i
                    pt, ps, _ = merged[i - 1]
                    new_text = pt + " " + text_i
                    new_start = ps
                    new_end = end_i
                    merged[i - 1] = (new_text, new_start, new_end)
                    embeddings_cache[i - 1] = _embed([new_text])[0]
                    merged.pop(i)
                    np.delete(embeddings_cache, i, axis=0)
                    i = max(0, i - 1)

                def _merge_with_next():
                    nt, _, ne = merged[i + 1]
                    new_text = text_i + " " + nt
                    new_start = start_i
                    new_end = ne
                    merged[i + 1] = (new_text, new_start, new_end)
                    embeddings_cache[i + 1] = _embed([new_text])[0]
                    merged.pop(i)
                    np.delete(embeddings_cache, i, axis=0)

                if has_prev and has_next:
                    current_emb = embeddings_cache[i: i + 1]
                    neighbors_emb = embeddings_cache[[i - 1, i + 1]]
                    sims = sklearn_cosine(current_emb, neighbors_emb)[0]
                    if sims[0] >= sims[1]:
                        _merge_with_prev()
                    else:
                        _merge_with_next()
                elif has_prev:
                    _merge_with_prev()
                else:
                    _merge_with_next()
            else:
                i += 1

        return merged

    # ------------------------------------------------------------------ #
    # Step 5 — Final size adjustment (chonkie SDPMChunker / SentenceChunker
    # could be used here, but a simple recursive sentence split is clearest)
    # ------------------------------------------------------------------ #

    def final_size_adjustment(
        chunks: List[tuple[str, int, int]],
    ) -> List[tuple[str, int, int]]:
        from chonkie import RecursiveChunker  # pure-text, no embeddings needed

        splitter = RecursiveChunker(chunk_size=final_threshold, chunk_overlap=0)

        result: List[tuple[str, int, int]] = []

        for chunk_text, chunk_start, chunk_end in chunks:
            if len(chunk_text) > final_threshold:
                sub_chunks = splitter.chunk(chunk_text)
                for sc in sub_chunks:
                    if not sc.text.strip():
                        continue
                    if hasattr(sc, "start_index") and sc.start_index is not None:
                        abs_start = chunk_start + sc.start_index
                        abs_end = chunk_start + sc.end_index
                    else:
                        abs_start, abs_end = _locate(sc.text, chunk_start)
                    result.append((sc.text, abs_start, abs_end))
            else:
                result.append((chunk_text, chunk_start, chunk_end))

        return result

    # ------------------------------------------------------------------ #
    # PIPELINE
    # ------------------------------------------------------------------ #

    final_tuples: List[tuple[str, int, int]] = []

    segments = segment_file(text)

    for seg_text, seg_start, seg_end in segments:

        initial_chunks = semantic_split(
            seg_text,
            initial_breakpoint_threshold,
            search_from=seg_start,
        )

        recursed: List[tuple[str, int, int]] = []

        for chunk_text, chunk_start, chunk_end in initial_chunks:
            if len(chunk_text) > recursive_threshold:
                recursed.extend(
                    recursive_semantic_split(
                        chunk_text,
                        chunk_start,
                        float(recursive_threshold),
                    )
                )
            else:
                recursed.append((chunk_text, chunk_start, chunk_end))

        merged = merge_small_chunks(recursed)
        final_tuples.extend(merged)

    final_tuples = final_size_adjustment(final_tuples)

    return [
        Chunk(chunk_text.strip(), start, end)
        for chunk_text, start, end in final_tuples
        if chunk_text.strip()
    ]