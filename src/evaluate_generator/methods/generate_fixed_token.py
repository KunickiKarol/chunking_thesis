import os
import time as time_module
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple
from vllm import LLM  # lazy import — not needed for API mode
from vllm import SamplingParams
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from src.evaluate_generator.methods.register import register_evaluator_generator
from src.evaluate_generator.prompts.all_prompts import generate_prompt
from src.tools.extract_llm import extract_verdict
from src.tools.tokenizer_service import TokenizerService


# ---------------------------------------------------------------------------
# Backend abstraction (z dokumentu + lazy singleton per wywołanie generate_fixed_token)
# ---------------------------------------------------------------------------


class _InferenceBackend(ABC):
    @abstractmethod
    def generate(self, document_block: str, max_new_tokens: int) -> str:
        pass


def _build_messages_from_parts(system_prompt: str, user_content: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


class _ApiBackend(_InferenceBackend):
    def __init__(self, model_id: str, base_url: str, api_key: str) -> None:
        

        self._model_id = model_id
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, system_prompt: str, user_content: str, max_new_tokens: int) -> str:
        messages = _build_messages_from_parts(system_prompt, user_content)
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()

class _OfflineBackend(_InferenceBackend):
    """
    vllm offline inference backend (in-process, no HTTP server required).
    Uses vllm.LLM + SamplingParams directly.
    """
    def __init__(self, model_id: str, hf_token: str | None) -> None:

        self._SamplingParams = SamplingParams
        self._llm = LLM(
        model=model_id,
        tokenizer=model_id,
        trust_remote_code=True,
        # Pass HF token via env if provided; vllm picks it up automatically
                )
        if hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    def generate(self, system_prompt, user_content, max_new_tokens) -> str:
        from vllm import SamplingParams
        messages = _build_messages_from_parts(system_prompt, user_content)
        # vllm.LLM.chat() accepts the OpenAI messages format directly
        outputs = self._llm.chat(
        messages=[messages],
        sampling_params=SamplingParams(
                            max_tokens=max_new_tokens,
                            temperature=0.0,
                                        ),
        use_tqdm=False,
                )
        return outputs[0].outputs[0].text.strip()


class _HuggingFaceBackend(_InferenceBackend):
    def __init__(self, model_id: str, hf_token: Optional[str]) -> None:


        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", token=hf_token, trust_remote_code=True
        )
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

    def generate(self, system_prompt: str, user_content: str, max_new_tokens: int) -> str:
        messages = _build_messages_from_parts(system_prompt, user_content)
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    class _HuggingFaceBackend(_InferenceBackend):
        def __init__(self, model_id: str, hf_token: Optional[str]) -> None:

            self._torch = torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                trust_remote_code=True,
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )

            self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        def generate(
            self,
            system_prompt: str,
            user_content: str,
            max_new_tokens: int,
        ) -> str:

            messages = _build_messages_from_parts(
                system_prompt,
                user_content,
            )

            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self._model.device)

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

            # tylko nowo wygenerowane tokeny
            generated_tokens = outputs[0][inputs.input_ids.shape[1] :]

            text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            return text


def _build_backend(params: dict) -> _InferenceBackend:
    mode = params.get("inference_mode", "api")
    model_id = params["model"]
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
# Singleton cache — model ładuje się raz na całe wywołanie generate_fixed_token
# ---------------------------------------------------------------------------

_BACKEND_CACHE: Dict[str, _InferenceBackend] = {}


def _get_or_build_backend(params: dict) -> _InferenceBackend:
    """
    Klucz cache = (inference_mode, model).
    Dzięki temu model jest ładowany tylko raz nawet gdy generate_answer
    jest wołane w pętli dla wielu pytań.
    """
    cache_key = f"{params.get('inference_mode', 'api')}::{params['model']}"
    if cache_key not in _BACKEND_CACHE:
        _BACKEND_CACHE[cache_key] = _build_backend(params)
    return _BACKEND_CACHE[cache_key]

# ---------------------------------------------------------------------------
# Główna funkcja generowania odpowiedzi dla jednego pytania
# ---------------------------------------------------------------------------

# Uzasadnienie podziału system/user w generate_prompt:
# generate_prompt zwraca (system_prompt, user_content) jako krotkę —
# dzięki temu możemy osobno policzyć tokeny overhead'u systemowego
# i dynamicznie dobierać chunki tylko do części "user".
# Jeśli generate_prompt zwraca str, traktujemy całość jako user_content
# z pustym system promptem.


def evaluate_final_answer(
    question_text: str,
    question_options: Optional[List[str]],
    gold_answers_text: List[str],
    mapped_answer: str,
    generation_preset_params: dict,
):
    """
    Ewaluje odpowiedź modelu dla pojedynczego pytania.

    Kroki:
    1. Pobierz/zbuduj tokenizer i backend (z cache).
    2. Zbuduj finalny prompt i wyślij do backendu.
    """
    params = generation_preset_params

    backend = _get_or_build_backend(params)

    # --- Zbuduj finalny prompt z wybranymi chunkami ------------------
    final_prompt = generate_prompt(
        params.get("prompt_template", "default_prompt"),
        question_text,
        gold_answers_text,
        mapped_answer,
    )

    if isinstance(final_prompt, tuple):
        system_prompt, user_content = final_prompt
    else:
        system_prompt = ""
        user_content = final_prompt

    # max_new_tokens — ile tokenów może wygenerować model
    # Zostawiamy rozsądny bufor; można parametryzować osobno
    max_new_tokens: int = params.get("max_new_tokens", 512)

    raw = backend.generate(system_prompt, user_content, max_new_tokens)
    return {"raw": raw, "verdict": extract_verdict(raw)}

# ---------------------------------------------------------------------------
# Funkcja batch — model ładuje się raz, pętla po pytaniach
# ---------------------------------------------------------------------------
@register_evaluator_generator("generate_fixed_token")
def generate_fixed_token(
    generation_results,
    tasks: Dict[str, dict],
    evaluation_preset_params: dict,
):
    """
    Generuje odpowiedzi dla wszystkich pytań.
    Backend (model) jest inicjalizowany raz dzięki _get_or_build_backend.
    """
    # Wymuś wczesną inicjalizację backendu i tokenizera —
    # żeby czas ładowania nie wchodził do mierzonych czasów odpowiedzi.
    _get_or_build_backend(evaluation_preset_params)

    results = {}
    total_time = 0.0

    for generation_task_id, generation_result in tqdm(generation_results.items(), desc="Evaluating generator"):
        task_data = tasks.get(generation_task_id)

        if not task_data:
            raise ValueError(f"Brak danych zadania dla ID {generation_task_id}")

        question_text = task_data["Question"]
        question_options = task_data.get("Options")
        gold_answers_text = task_data["Answers"]
        answer = generation_result["extracted"]
        mapped_answer = extract_verdict(answer)
        start_time = time_module.perf_counter()
        metrics = evaluate_final_answer(
            question_text,
            question_options,
            gold_answers_text,
            mapped_answer,
            evaluation_preset_params
        )
        answer_time = time_module.perf_counter() - start_time

        total_time += answer_time
        results[generation_task_id] = metrics

    return results, total_time
