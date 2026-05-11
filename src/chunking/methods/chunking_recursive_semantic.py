from typing import List, Optional
import re
import math

from langchain_experimental.text_splitter import SemanticChunker
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import nltk
import numpy as np
import torch

from src.chunking.methods.register import register_chunker


@register_chunker("recursive_semantic")
def chunking_semantic_recursive(
    text: str,
    **params,
) -> List[str]:
    """
    Recursive Semantic Chunking.

    Returns:
        List[str] -> final chunks
    """
    max_chunk_size = params.get("max_chunk_size")
    recursive_threshold = params.get("recursive_threshold")
    final_threshold = params.get("final_threshold")
    merge_threshold = params.get("merge_threshold")
    delta = params.get("delta")
    embedding_model = params.get("embedding_model")
    breakpoint_threshold_type = params.get("breakpoint_threshold_type",)
    initial_breakpoint_threshold = params.get("initial_breakpoint_threshold",)
    embeddings = params.get("embeddings")

    if not text or not text.strip():
        return []

    if delta <= 0:
        raise ValueError("delta must be positive")

    if merge_threshold >= recursive_threshold:
        raise ValueError(
            "merge_threshold must be smaller than recursive_threshold"
        )

    if final_threshold > max_chunk_size:
        raise ValueError(
            "final_threshold must not exceed max_chunk_size"
        )
    nltk.download('punkt_tab')
    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},   # <- GPU / CPU
            encode_kwargs={"device": device},   # <- ważne dla embeddingów
        )

    semantic_splitter_cache = {}

    def get_semantic_splitter(breakpoint_threshold: float):
        key = round(breakpoint_threshold, 6)

        if key not in semantic_splitter_cache:
            semantic_splitter_cache[key] = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold,
            )

        return semantic_splitter_cache[key]

    # ------------------------------------------------------------------ #
    # Sentence tokenizer
    # ------------------------------------------------------------------ #

    def tokenize_sentences(text_: str) -> List[str]:
        try:
            import nltk

            try:
                return nltk.sent_tokenize(text_)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                return nltk.sent_tokenize(text_)

        except ImportError:
            return [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", text_)
                if s.strip()
            ]

    # ------------------------------------------------------------------ #
    # Step 1 — Segment file
    # ------------------------------------------------------------------ #

    def segment_file(text_: str) -> List[str]:
        if len(text_) <= max_chunk_size:
            return [text_]

        sentences = tokenize_sentences(text_)

        segments = []
        current_parts = []
        current_len = 0

        for sentence in sentences:
            slen = len(sentence)

            # giant sentence fallback
            if slen > max_chunk_size:
                if current_parts:
                    segments.append(" ".join(current_parts))
                    current_parts = []
                    current_len = 0

                for start in range(0, slen, max_chunk_size):
                    segments.append(
                        sentence[start:start + max_chunk_size]
                    )

                continue

            if current_len + slen > max_chunk_size:
                segments.append(" ".join(current_parts))
                current_parts = []
                current_len = 0

            current_parts.append(sentence)
            current_len += slen + 1

        if current_parts:
            segments.append(" ".join(current_parts))

        return segments

    # ------------------------------------------------------------------ #
    # Step 2 — Semantic split
    # ------------------------------------------------------------------ #

    def semantic_split(
        text_: str,
        breakpoint_threshold: float,
    ) -> List[str]:

        splitter = get_semantic_splitter(breakpoint_threshold)

        docs = splitter.create_documents([text_])

        return [
            d.page_content
            for d in docs
            if d.page_content.strip()
        ]

    # ------------------------------------------------------------------ #
    # Step 3 — Recursive semantic split
    # ------------------------------------------------------------------ #

    def recursive_semantic_split(
        chunk: str,
        threshold: float,
    ) -> List[str]:

        if len(chunk) <= threshold:
            return [chunk]

        next_threshold = max(threshold - delta, 1.0)

        sub_chunks = semantic_split(
            chunk,
            initial_breakpoint_threshold,
        )

        # no progress fallback
        if (
            len(sub_chunks) == 1
            and len(sub_chunks[0]) >= len(chunk)
        ):
            return [chunk]

        result = []

        for sub_chunk in sub_chunks:
            result.extend(
                recursive_semantic_split(
                    sub_chunk,
                    next_threshold,
                )
            )

        return result

    # ------------------------------------------------------------------ #
    # Step 4 — Merge small chunks
    # ------------------------------------------------------------------ #

    def embed_texts(texts: List[str]):
        return embeddings.embed_documents(texts)


    def merge_small_chunks(chunks: List[str]) -> List[str]:
        """Łączy małe chunki z sąsiadami na podstawie similarity (zoptymalizowane)"""
        
        if len(chunks) <= 1:
            return chunks

        merged = list(chunks)
        
        # Embeduj wszystkie chunki na raz i konwertuj na numpy array
        embeddings_cache = np.array(embed_texts(merged))  # Shape: (N, embedding_dim)

        i = 0
        while i < len(merged):
            if (
                len(merged[i]) <= merge_threshold
                and len(merged) > 1
            ):
                has_prev = i > 0
                has_next = i < len(merged) - 1

                if has_prev and has_next:
                    # Oblicz similarity dla dwóch par na raz (batch)
                    current_emb = embeddings_cache[i:i+1]  # Shape: (1, dim)
                    neighbors_emb = embeddings_cache[[i-1, i+1]]  # Shape: (2, dim)
                    
                    similarities = sklearn_cosine(current_emb, neighbors_emb)[0]  # Shape: (2,)
                    sim_prev, sim_next = similarities[0], similarities[1]

                    if sim_prev >= sim_next:
                        # Merge z poprzednim
                        merged[i - 1] += " " + merged[i]
                        
                        # Re-embed tylko zmieniony chunk
                        embeddings_cache[i - 1] = np.array(
                            embed_texts([merged[i - 1]])[0]
                        )
                        
                        merged.pop(i)
                        embeddings_cache = np.delete(embeddings_cache, i, axis=0)
                        i = max(0, i - 1)

                    else:
                        # Merge z następnym
                        merged[i + 1] = merged[i] + " " + merged[i + 1]
                        
                        embeddings_cache[i + 1] = np.array(
                            embed_texts([merged[i + 1]])[0]
                        )
                        
                        merged.pop(i)
                        embeddings_cache = np.delete(embeddings_cache, i, axis=0)

                elif has_prev:
                    # Tylko poprzedni
                    merged[i - 1] += " " + merged[i]
                    
                    embeddings_cache[i - 1] = np.array(
                        embed_texts([merged[i - 1]])[0]
                    )
                    
                    merged.pop(i)
                    embeddings_cache = np.delete(embeddings_cache, i, axis=0)
                    i = max(0, i - 1)

                else:
                    # Tylko następny
                    merged[i + 1] = merged[i] + " " + merged[i + 1]
                    
                    embeddings_cache[i + 1] = np.array(
                        embed_texts([merged[i + 1]])[0]
                    )
                    
                    merged.pop(i)
                    embeddings_cache = np.delete(embeddings_cache, i, axis=0)

            else:
                i += 1

        return merged

    # ------------------------------------------------------------------ #
    # Step 5 — Final size adjustment
    # ------------------------------------------------------------------ #

    def final_size_adjustment(
        chunks: List[str],
    ) -> List[str]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=final_threshold,
            chunk_overlap=0,
            length_function=len,
        )

        result = []

        for chunk in chunks:

            if len(chunk) > final_threshold:

                docs = splitter.create_documents([chunk])

                result.extend(
                    d.page_content
                    for d in docs
                    if d.page_content.strip()
                )

            else:
                result.append(chunk)

        return result

    # ------------------------------------------------------------------ #
    # PIPELINE
    # ------------------------------------------------------------------ #

    final_chunks = []

    segments = segment_file(text)

    for segment in segments:

        initial_chunks = semantic_split(
            segment,
            initial_breakpoint_threshold,
        )

        recursed_chunks = []

        for chunk in initial_chunks:

            if len(chunk) > recursive_threshold:

                recursed_chunks.extend(
                    recursive_semantic_split(
                        chunk,
                        float(recursive_threshold),
                    )
                )

            else:
                recursed_chunks.append(chunk)

        merged_chunks = merge_small_chunks(
            recursed_chunks
        )

        final_chunks.extend(merged_chunks)

    final_chunks = final_size_adjustment(
        final_chunks
    )

    return [
        chunk.strip()
        for chunk in final_chunks
        if chunk.strip()
    ]