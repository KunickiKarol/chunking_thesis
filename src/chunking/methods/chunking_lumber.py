"""
LumberChunker with three interchangeable inference backends:

    inference_mode = "hf"       →  HuggingFace AutoModelForCausalLM (transformers)
    inference_mode = "offline"  →  vllm.LLM  (in-process, no server needed)
    inference_mode = "api"      →  OpenAI-compatible HTTP client
                                   (local vllm serve, RunPod, Modal, …)

Switch by passing  inference_mode=<"hf"|"offline"|"api">  in **params.
"""

from __future__ import annotations

import os
import re
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List

from nltk.tokenize import sent_tokenize

from src.chunking.methods.register import register_chunker


@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

    Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

    Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX', without any explanatory notes.

    Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""
)


def _build_messages(document_block: str) -> List[dict]:
    """Build the chat messages list sent to the LLM."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"\nDocument:\n{document_block}"},
    ]


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class _InferenceBackend(ABC):
    """Common interface for both inference backends."""

    @abstractmethod
    def generate(self, document_block: str, max_new_tokens: int) -> str:
        """Return the raw model reply for *document_block*."""


class _ApiBackend(_InferenceBackend):
    def __init__(self, model_id: str, base_url: str, api_key: str) -> None:
        from openai import OpenAI

        self._model_id = model_id
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, document_block: str, max_new_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=_build_messages(document_block),
            max_tokens=max_new_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()


class _HuggingFaceBackend(_InferenceBackend):
    def __init__(self, model_id: str, hf_token: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True,
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", token=hf_token,
        )

    def generate(self, document_block: str, max_new_tokens: int) -> str:
        messages = _build_messages(document_block)
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class _OfflineBackend(_InferenceBackend):
    def __init__(self, model_id: str, hf_token: str | None) -> None:
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self._llm = LLM(model=model_id, tokenizer=model_id, trust_remote_code=True)
        if hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    def generate(self, document_block: str, max_new_tokens: int) -> str:
        from vllm import SamplingParams

        messages = _build_messages(document_block)
        outputs = self._llm.chat(
            messages=[messages],
            sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _build_backend(params: dict) -> _InferenceBackend:
    mode = params.get("inference_mode", "api")
    model_id = params.get("model")
    hf_token = os.environ.get("HF_TOKEN")

    if mode == "hf":
        return _HuggingFaceBackend(model_id=model_id, hf_token=hf_token)
    if mode == "offline":
        return _OfflineBackend(model_id=model_id, hf_token=hf_token)
    if mode == "api":
        base_url = params.get("vllm_base_url") or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = params.get("vllm_api_key") or os.environ.get("VLLM_API_KEY", "token-abc123")
        return _ApiBackend(model_id=model_id, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unknown inference_mode '{mode}'. Choose 'hf', 'offline', or 'api'.")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def _llm_prompt(
    backend: _InferenceBackend,
    document_block: str,
    max_new_tokens: int,
    max_retries: int,
    sleep_seconds: int,
) -> str:
    last_exception = None
    for attempt in range(max_retries):
        try:
            return backend.generate(document_block, max_new_tokens)
        except Exception as exc:
            last_exception = exc
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError("LLM prompt failed after retries") from last_exception


def _count_words(text: str) -> int:
    return round(1.2 * len(text.split()))


# ---------------------------------------------------------------------------
# Character-level offset helpers
# ---------------------------------------------------------------------------


def _build_sentence_char_spans(text: str, sentences: List[str]) -> List[tuple[int, int]]:
    """
    For each sentence return (start_char, end_char) in the original *text*.

    We walk through *text* linearly, anchoring each sentence at the earliest
    occurrence at-or-after the previous sentence's end.  This correctly handles
    duplicate sentences anywhere in the document.
    """
    spans: List[tuple[int, int]] = []
    search_from = 0
    for sentence in sentences:
        start = text.find(sentence, search_from)
        if start == -1:
            # Fallback: should never happen with sent_tokenize output, but be safe.
            start = search_from
        end = start + len(sentence)
        spans.append((start, end))
        search_from = end
    return spans


# ---------------------------------------------------------------------------
# LumberChunker
# ---------------------------------------------------------------------------


@register_chunker("lumber")
def chunking_lumber(text: str, **params) -> List[Chunk]:
    """
    Split *text* into semantically coherent chunks using the LumberChunker method.

    Returns:
        List[Chunk] – each chunk carries .text, .start_index, .end_index
                      where start_index / end_index are character offsets
                      into the original *text*.
    """
    # --- required params ---
    model_id = params.get("model")
    group_size_threshold = int(params.get("group_size_threshold"))
    max_retries = int(params.get("max_retries"))
    sleep_seconds = int(params.get("sleep_seconds"))
    max_new_tokens = int(params.get("max_new_tokens"))

    if model_id is None:
        raise ValueError("'model' param is required")
    if group_size_threshold <= 0:
        raise ValueError("group_size_threshold must be positive")
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be non-negative")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    # --- build backend ---
    backend = _build_backend(params)

    # --- tokenise text into sentences ---
    sentences = sent_tokenize(text)

    # Pre-compute character spans for every sentence in the original text.
    # This is the single source of truth for start_index / end_index.
    sentence_spans: List[tuple[int, int]] = _build_sentence_char_spans(text, sentences)

    full_segments = [f"ID {idx}: {seg}" for idx, seg in enumerate(sentences)]

    # --- main LumberChunker loop ---
    chunk_number = 0
    boundary_ids: List[int] = []

    while chunk_number < len(full_segments) - 5:
        word_count = 0
        i = 0

        while word_count < group_size_threshold and (i + chunk_number) < len(full_segments) - 1:
            i += 1
            candidate = "\n".join(full_segments[k] for k in range(chunk_number, i + chunk_number))
            word_count = _count_words(candidate)

        if i == 1:
            final_document = "\n".join(full_segments[k] for k in range(chunk_number, i + chunk_number))
        else:
            final_document = "\n".join(full_segments[k] for k in range(chunk_number, i - 1 + chunk_number))

        chunk_number = chunk_number + i - 1

        try:
            llm_output = _llm_prompt(
                backend=backend,
                document_block=final_document,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        except RuntimeError as exc:
            print(f"LLM prompt failed for chunk starting at ID {chunk_number}. Error: {exc}")
            chunk_number += 1
            continue

        if llm_output == "content_flag_increment":
            chunk_number += 1
            continue

        match = re.search(r"Answer: ID \w+", llm_output)
        if match is None:
            chunk_number += 1
        else:
            id_match = re.search(r"\d+", match.group(0))
            chunk_number = int(id_match.group())
            boundary_ids.append(chunk_number)
            chunk_number += 1

    boundary_ids.append(len(full_segments))

    # --- assemble Chunk objects using pre-computed character spans ---
    chunks: List[Chunk] = []
    for i, end_idx in enumerate(boundary_ids):
        start_idx = boundary_ids[i - 1] if i > 0 else 0

        # Character offsets: first char of first sentence … last char of last sentence
        char_start = sentence_spans[start_idx][0]
        char_end = sentence_spans[end_idx - 1][1]  # end_idx is exclusive → -1

        chunk_text = text[char_start:char_end]
        if chunk_text.strip():
            chunks.append(Chunk(text=chunk_text, start_index=char_start, end_index=char_end))

    return chunks