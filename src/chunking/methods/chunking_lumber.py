"""
LumberChunker with three interchangeable inference backends:

    inference_mode = "hf"       →  HuggingFace AutoModelForCausalLM (transformers)
    inference_mode = "offline"  →  vllm.LLM  (in-process, no server needed)
    inference_mode = "api"      →  OpenAI-compatible HTTP client
                                   (local vllm serve, RunPod, Modal, …)

Switch by passing  inference_mode=<"hf"|"offline"|"api">  in **params.
"""

from __future__ import annotations

import re
import time
import textwrap
import os
from abc import ABC, abstractmethod
from typing import List, Callable, Dict

from nltk.tokenize import sent_tokenize

from src.chunking.methods.register import register_chunker


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

    Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

    Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX', without any explanatory notes.

    Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""
)


def _build_messages(document_block: str) -> List[dict]:
    """Build the chat messages list sent to the LLM."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": f"\nDocument:\n{document_block}"},
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
    """
    OpenAI-compatible HTTP backend (vllm serve, RunPod, Modal, …).

    Switch between local and remote by changing *base_url*:
        local  → "http://localhost:8000/v1"
        remote → "https://<your-host>/v1"
    """

    def __init__(self, model_id: str, base_url: str, api_key: str) -> None:
        from openai import OpenAI  # lazy import — not needed for offline mode
        self._model_id = model_id
        self._client   = OpenAI(base_url=base_url, api_key=api_key)

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
    """
    HuggingFace transformers backend using AutoModelForCausalLM.

    Loads the model directly into the current process via the transformers library.
    Suitable when vllm is not available or for CPU-only environments.
    """

    def __init__(self, model_id: str, hf_token: str | None) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM  # lazy imports

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            token=hf_token,
        )

    def generate(self, document_block: str, max_new_tokens: int) -> str:
        messages = _build_messages(document_block)

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class _OfflineBackend(_InferenceBackend):
    """
    vllm offline inference backend (in-process, no HTTP server required).

    Uses vllm.LLM + SamplingParams directly.
    """

    def __init__(self, model_id: str, hf_token: str | None) -> None:
        from vllm import LLM, SamplingParams  # lazy import — not needed for API mode
        self._SamplingParams = SamplingParams
        self._llm = LLM(
            model=model_id,
            tokenizer=model_id,
            trust_remote_code=True,
            # Pass HF token via env if provided; vllm picks it up automatically
        )
        if hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    def generate(self, document_block: str, max_new_tokens: int) -> str:
        from vllm import SamplingParams
        messages   = _build_messages(document_block)
        # vllm.LLM.chat() accepts the OpenAI messages format directly
        outputs    = self._llm.chat(
            messages=[messages],
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
            ),
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(params: dict) -> _InferenceBackend:
    """
    Instantiate the correct backend from *params*.

    params["inference_mode"] controls which backend is used:
        "hf"       →  _HuggingFaceBackend  (transformers AutoModelForCausalLM)
        "offline"  →  _OfflineBackend      (vllm.LLM, in-process)
        "api"      →  _ApiBackend          (OpenAI-compatible HTTP)
    """
    mode     = params.get("inference_mode", "api")
    model_id = params.get("model")
    hf_token = os.environ.get("HF_TOKEN")

    if mode == "hf":
        return _HuggingFaceBackend(model_id=model_id, hf_token=hf_token)

    if mode == "offline":
        return _OfflineBackend(model_id=model_id, hf_token=hf_token)

    if mode == "api":
        base_url = (
            params.get("vllm_base_url")
            or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        api_key = (
            params.get("vllm_api_key")
            or os.environ.get("VLLM_API_KEY", "token-abc123")
        )
        return _ApiBackend(model_id=model_id, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unknown inference_mode '{mode}'. Choose 'hf', 'offline', or 'api'.")


# ---------------------------------------------------------------------------
# Retry wrapper (backend-agnostic)
# ---------------------------------------------------------------------------

def _llm_prompt(
    backend: _InferenceBackend,
    document_block: str,
    max_new_tokens: int,
    max_retries: int,
    sleep_seconds: int,
) -> str:
    """
    Call *backend.generate()* with retry logic.
    Raises RuntimeError after exhausting all retries.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return backend.generate(document_block, max_new_tokens)
        except Exception as exc:
            last_exception = exc
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError("LLM prompt failed after retries") from last_exception


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    """Approximate token count: 1 word ≈ 1.2 tokens."""
    return round(1.2 * len(text.split()))


# ---------------------------------------------------------------------------
# LumberChunker
# ---------------------------------------------------------------------------

@register_chunker("lumber")
def chunking_lumber(text: str, **params) -> List[str]:
    """
    Split *text* into semantically coherent chunks using the LumberChunker method.

    Required params:
        model                – HuggingFace model ID
        group_size_threshold – max approximate token count per candidate group
        max_retries          – LLM call retries on failure
        sleep_seconds        – seconds to wait between retries
        max_new_tokens       – max tokens the LLM may generate per call

    Backend selection:
        inference_mode = "hf"       – transformers AutoModelForCausalLM (no vllm needed)
            (no extra params needed; model is loaded directly via HuggingFace)

        inference_mode = "offline"  – vllm.LLM in-process (no server)
            (no extra params needed; model is loaded directly)

        inference_mode = "api"      (default) – OpenAI-compatible HTTP client
            vllm_base_url  – server URL  (env: VLLM_BASE_URL, default: http://localhost:8000/v1)
            vllm_api_key   – API key     (env: VLLM_API_KEY,  default: token-abc123)

    Environment variables:
        HUGGINGFACEHF_TOKEN_HUB_TOKEN – HuggingFace Hub token
        VLLM_BASE_URL                 – fallback for vllm_base_url
        VLLM_API_KEY                  – fallback for vllm_api_key

    Returns:
        List[str] – the resulting text chunks.
    """
    # --- required params (no defaults — caller must supply all five) ---
    model_id             = params.get("model")
    group_size_threshold = int(params.get("group_size_threshold"))
    max_retries          = int(params.get("max_retries"))
    sleep_seconds        = int(params.get("sleep_seconds"))
    max_new_tokens       = int(params.get("max_new_tokens"))

    # --- validation ---
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

    # --- build backend (offline or api) ---
    backend = _build_backend(params)

    # --- tokenise text into sentences and label with IDs ---
    sentences     = sent_tokenize(text)
    full_segments = [f"ID {idx}: {seg}" for idx, seg in enumerate(sentences)]

    # --- main LumberChunker loop ---
    chunk_number  = 0
    boundary_ids: List[int] = []

    while chunk_number < len(full_segments) - 5:
        word_count = 0
        i = 0

        while word_count < group_size_threshold and (i + chunk_number) < len(full_segments) - 1:
            i += 1
            candidate  = "\n".join(full_segments[k] for k in range(chunk_number, i + chunk_number))
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
            id_match     = re.search(r"\d+", match.group(0))
            chunk_number = int(id_match.group())
            boundary_ids.append(chunk_number)
            chunk_number += 1

    boundary_ids.append(len(full_segments))

    clean_segments = [re.sub(r"^ID \d+:\s*", "", seg) for seg in full_segments]

    chunks: List[str] = []
    for i, end_idx in enumerate(boundary_ids):
        start_idx  = boundary_ids[i - 1] if i > 0 else 0
        chunk_text = "\n".join(clean_segments[start_idx:end_idx])
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks
