from typing import List

from chonkie import NeuralChunker, TokenChunker
from transformers import AutoTokenizer

from src.chunking.methods.register import register_chunker


@register_chunker("neural")
def chunking_neural(
    text: str,
    **params,
) -> List[str]:
    """
    Hybrid chunking:
    1. TokenChunker (fixed-size token splitting)
    2. NeuralChunker (semantic refinement)
    """

    neural_model = params.get("neural_model")
    min_characters_per_chunk = params.get("min_characters_per_chunk")
    chunk_size = params.get("chunk_size")
    overlap = params.get("overlap")

    # =========================================================
    # 1. TOKENIZER (from model)
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        neural_model,
        use_fast=True,
    )

    # =========================================================
    # 2. FIXED-SIZE TOKEN CHUNKER (NO MANUAL WINDOWING)
    # =========================================================
    token_chunker = TokenChunker(
        tokenizer, chunk_size=chunk_size, chunk_overlap=overlap  # safe under ModernBERT 8192 limit
    )

    # raw fixed-size chunks
    pre_chunks = token_chunker(text)

    # =========================================================
    # 3. NEURAL REFINEMENT LAYER
    # =========================================================
    neural_chunker = NeuralChunker(model=neural_model, min_characters_per_chunk=min_characters_per_chunk)

    final_chunks: List[str] = []

    for chunk in pre_chunks:

        # =========================================================
        # DEBUG: token count BEFORE NeuralChunker
        # =========================================================
        token_count = len(tokenizer.encode(chunk.text, add_special_tokens=False))

        # optional safety check (helps catch leaks)
        if token_count > 1024:
            print("[WARNING] chunk exceeds model limit!")

        try:
            refined = neural_chunker.chunk(chunk.text)
            final_chunks.extend([c.text for c in refined if c.text.strip()])
        except Exception:
            if chunk.text.strip():
                final_chunks.append(chunk.text)

    return final_chunks
