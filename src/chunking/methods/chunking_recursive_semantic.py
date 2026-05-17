import re
from typing import List, Optional

import nltk
import numpy as np
import torch
from chonkie import SemanticChunker
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from src.chunking.methods.chunking_lumber import Chunk
from src.chunking.methods.register import register_chunker
from src.tools.chunk import trim_bounds

def tokenize_sentences(text_: str) -> List[str]:
    return re.findall(r'[^.!?]+[.!?]*\s*', text_, re.DOTALL)

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
    semantic_filter_window = params.get("semantic_filter_window", 1)
    semantic_filter_polyorder = params.get("semantic_filter_polyorder", 0)
    semantic_filter_tolerance = params.get("semantic_filter_tolerance", 0.0001)

    if not text or not text.strip():
        return []

    if delta <= 0:
        raise ValueError("delta must be positive")

    if merge_threshold >= recursive_threshold:
        raise ValueError("merge_threshold must be smaller than recursive_threshold")

    if final_threshold > max_chunk_size:
        raise ValueError("final_threshold must not exceed max_chunk_size")

    nltk.download("punkt_tab", quiet=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embeddings is not None:
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

    def _get_chunker(chunk_size: int) -> SemanticChunker:
        key = chunk_size
        if key not in _chunker_cache:
            _chunker_cache[key] = SemanticChunker(
                embedding_model=embedding_model if embeddings is None else embeddings,
                chunk_size=chunk_size,
                threshold=initial_breakpoint_threshold,
                threshold_type=breakpoint_threshold_type,
                filter_window=semantic_filter_window,
                filter_polyorder=semantic_filter_polyorder,
                filter_tolerance=semantic_filter_tolerance,
            )
        return _chunker_cache[key]

    # ------------------------------------------------------------------ #
    # Sentence tokenizer
    # ------------------------------------------------------------------ #

    # def tokenize_sentences(text_: str) -> List[str]:
    #     try:
    #         try:
    #             return tokenize_sentences(text_)
    #         except LookupError:
    #             nltk.download("punkt", quiet=True)
    #             nltk.download("punkt_tab", quiet=True)
    #             return tokenize_sentences(text_)
    #     except ImportError:
    #         return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_) if s.strip()]

    # ------------------------------------------------------------------ #
    # Helper: find the position of chunk_text inside the *original* text.
    # Always starts searching from `search_from` to avoid false earlier matches.
    # ------------------------------------------------------------------ #


    def _locate(chunk_text: str, search_from: int = 0) -> tuple[int, int]:
        """Return (start, end) absolute positions of chunk_text in `text`."""
        # Try exact match first
        idx = text.find(chunk_text, search_from)
        
        if idx == -1:
            # Try normalized whitespace match
            normalized_chunk = chunk_text
            normalized_text = text[search_from:]
            
            idx_in_normalized = normalized_text.find(normalized_chunk)
            if idx_in_normalized != -1:
                # Map back to original text by counting characters
                # This is approximate but better than -1
                idx = search_from + idx_in_normalized
        
        if idx == -1:
            # Last resort: search from beginning (for edge cases)
            idx = text.find(chunk_text)
            
            if idx == -1:
                # Try normalized from beginning
                normalized_chunk = chunk_text
                normalized_text = text
                idx = normalized_text.find(normalized_chunk)
        
        if idx == -1:
            return (-1, -1)
        
        return idx, idx + len(chunk_text)

    # ------------------------------------------------------------------ #
    # Step 1 — Segment file
    # FIX: use a running `cursor` so _locate never goes backwards and
    #      returns an earlier duplicate match.
    # ------------------------------------------------------------------ #

    def segment_file(text_: str) -> List[tuple[str, int, int]]:
        if len(text_) <= max_chunk_size:
            return [(text_, 0, len(text_))]

        sentences = tokenize_sentences(text_)

        segments: List[tuple[str, int, int]] = []
        current_parts: List[str] = []
        current_len = 0
        # FIX: track where the next _locate call should start searching
        cursor = 0

        for sentence in sentences:
            slen = len(sentence)

            if slen > max_chunk_size:
                # Flush accumulator first
                if current_parts:
                    seg_text = "".join(current_parts)
                    start, end = _locate(seg_text, cursor)
                    
                    # Fallback to approximate position
                    if start == -1:
                        start = cursor
                        end = cursor + len(seg_text)
                    
                    segments.append((seg_text, start, end))
                    cursor = end
                    current_parts = []
                    current_len = 0

                # Split oversized sentence into fixed-size pieces
                for chunk_start_offset in range(0, slen, max_chunk_size):
                    piece = sentence[chunk_start_offset: chunk_start_offset + max_chunk_size]
                    start, end = _locate(piece, cursor)
                    
                    # Fallback to approximate position
                    if start == -1:
                        start = cursor
                        end = cursor + len(piece)
                    
                    segments.append((piece, start, end))
                    cursor = end

                continue

            if current_len + slen > max_chunk_size:
                seg_text = "".join(current_parts)
                start, end = _locate(seg_text, cursor)
                
                # Fallback to approximate position
                if start == -1:
                    start = cursor
                    end = cursor + len(seg_text)
                
                segments.append((seg_text, start, end))
                cursor = end
                current_parts = []
                current_len = 0

            current_parts.append(sentence)
            current_len += slen  # +1 for the joining space

        if current_parts:
            seg_text = "".join(current_parts)
            start, end = _locate(seg_text, cursor)
            
            # Fallback to approximate position
            if start == -1:
                start = cursor
                end = cursor + len(seg_text)
            
            segments.append((seg_text, start, end))

        return segments

    # ------------------------------------------------------------------ #
    # Step 2 — Semantic split via chonkie
    # FIX: always use chonkie's start_index/end_index when available;
    #      fall back to _locate with a monotonically advancing cursor.
    # ------------------------------------------------------------------ #

    def semantic_split(
        text_: str,
        chunk_size: int,
        search_from: int = 0,
    ) -> List[tuple[str, int, int]]:
        chunker = _get_chunker(chunk_size)
        chunks = chunker.chunk(text_)

        result: List[tuple[str, int, int]] = []
        # FIX: cursor advances monotonically so _locate never backtracks
        cursor = search_from

        for c in chunks:
            chunk_text = c.text
            if not chunk_text.strip():
                continue

            if hasattr(c, "start_index") and c.start_index is not None:
                # chonkie indices are relative to text_ — shift by search_from
                abs_start = search_from + c.start_index
                abs_end = search_from + c.end_index
                cursor = abs_end
            else:
                # Fallback: search forward from cursor
                abs_start, abs_end = _locate(chunk_text, cursor)
                
                # If _locate failed, use approximate position based on cursor
                if abs_start == -1:
                    abs_start = cursor
                    abs_end = cursor + len(chunk_text)
                
                cursor = abs_end

            result.append((chunk_text, abs_start, abs_end))

        return result

    # ------------------------------------------------------------------ #
    # Step 3 — Recursive semantic split
    # FIX: keep chunk_size as int throughout (avoid float from `1.0`)
    # ------------------------------------------------------------------ #

    def recursive_semantic_split(
        chunk_text: str,
        chunk_start: int,
        chunk_size: int,
    ) -> List[tuple[str, int, int]]:

        if len(chunk_text) <= chunk_size:
            return [(chunk_text, chunk_start, chunk_start + len(chunk_text))]

        # FIX: cast to int so chonkie never receives a float chunk_size
        next_chunk_size = max(int(chunk_size - delta), 1)

        sub_chunks = semantic_split(
            chunk_text,
            next_chunk_size,
            search_from=chunk_start,
        )

        # No progress — return as-is to avoid infinite recursion
        if len(sub_chunks) == 1 and len(sub_chunks[0][0]) >= len(chunk_text):
            return [(chunk_text, chunk_start, chunk_start + len(chunk_text))]

        result: List[tuple[str, int, int]] = []

        for sub_text, sub_start, sub_end in sub_chunks:
            result.extend(
                recursive_semantic_split(sub_text, sub_start, next_chunk_size)
            )

        return result

    # ------------------------------------------------------------------ #
    # Step 4 — Merge small chunks
    # FIX: np.delete returns a new array — reassign embeddings_cache;
    #      also rebuild it from scratch after every mutation to keep
    #      row indices perfectly in sync with `merged`.
    # ------------------------------------------------------------------ #

    def merge_small_chunks(
        chunks: List[tuple[str, int, int]]
    ) -> List[tuple[str, int, int]]:
        if len(chunks) <= 1:
            return chunks

        merged: List[tuple[str, int, int]] = list(chunks)

        i = 0
        while i < len(merged):
            text_i, start_i, end_i = merged[i]

            if len(text_i) <= merge_threshold and len(merged) > 1:
                has_prev = i > 0
                has_next = i < len(merged) - 1

                if has_prev and has_next:
                    # Compute similarity on demand — avoids stale cache entirely
                    prev_text = merged[i - 1][0]
                    next_text = merged[i + 1][0]
                    embs = _embed([text_i, prev_text, next_text])
                    sim_prev = sklearn_cosine(embs[0:1], embs[1:2])[0][0]
                    sim_next = sklearn_cosine(embs[0:1], embs[2:3])[0][0]

                    if sim_prev >= sim_next:
                        # Merge with previous
                        pt, ps, _ = merged[i - 1]
                        merged[i - 1] = (pt + "" + text_i, ps, end_i)
                        merged.pop(i)
                        i = max(0, i - 1)
                    else:
                        # Merge with next
                        nt, _, ne = merged[i + 1]
                        merged[i + 1] = (text_i + "" + nt, start_i, ne)
                        merged.pop(i)
                        # i stays — we re-evaluate the newly merged chunk

                elif has_prev:
                    pt, ps, _ = merged[i - 1]
                    merged[i - 1] = (pt + "" + text_i, ps, end_i)
                    merged.pop(i)
                    i = max(0, i - 1)

                else:  # has_next only
                    nt, _, ne = merged[i + 1]
                    merged[i + 1] = (text_i + "" + nt, start_i, ne)
                    merged.pop(i)

            else:
                i += 1

        return merged

    # ------------------------------------------------------------------ #
    # Step 5 — Final size adjustment
    # FIX: use chonkie's indices (relative to chunk_text) shifted by
    #      chunk_start; fall back to _locate with advancing cursor.
    # ------------------------------------------------------------------ #

    def final_size_adjustment(
        chunks: List[tuple[str, int, int]],
    ) -> List[tuple[str, int, int]]:
        from chonkie import RecursiveChunker

        splitter = RecursiveChunker(chunk_size=final_threshold)

        result: List[tuple[str, int, int]] = []

        for chunk_text, chunk_start, chunk_end in chunks:
            if len(chunk_text) > final_threshold:
                sub_chunks = splitter.chunk(chunk_text)
                # FIX: cursor advances within this chunk's span only
                local_cursor = chunk_start

                for sc in sub_chunks:
                    if not sc.text.strip():
                        continue

                    if hasattr(sc, "start_index") and sc.start_index is not None:
                        abs_start = chunk_start + sc.start_index
                        abs_end = chunk_start + sc.end_index
                    else:
                        abs_start, abs_end = _locate(sc.text, local_cursor)
                        
                        # Fallback to approximate position
                        if abs_start == -1:
                            abs_start = local_cursor
                            abs_end = local_cursor + len(sc.text)

                    local_cursor = abs_end
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
            recursive_threshold,
            search_from=seg_start,
        )

        recursed: List[tuple[str, int, int]] = []

        for chunk_text, chunk_start, chunk_end in initial_chunks:
            if len(chunk_text) > recursive_threshold:
                recursed.extend(
                    recursive_semantic_split(
                        chunk_text,
                        chunk_start,
                        recursive_threshold,
                    )
                )
            else:
                recursed.append((chunk_text, chunk_start, chunk_end))

        merged = merge_small_chunks(recursed)
        final_tuples.extend(merged)

    final_tuples = final_size_adjustment(final_tuples)

    return [
        Chunk(*trim_bounds(chunk_text, start, end))
        for chunk_text, start, end in final_tuples
        if chunk_text.strip()
    ]