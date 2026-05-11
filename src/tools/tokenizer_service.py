# tokenizer_service.py

from typing import Dict, Any

from transformers import AutoTokenizer
import yaml


try:
    import tiktoken
except ImportError:
    tiktoken = None


class TokenizerService:

    def __init__(
        self,
        backend: str = None,
        model_name: str = None,
    ):

        # =========================
        # DEFAULT VALUES
        # =========================

        default_backend = "tiktoken"
        default_model_name = "gpt-4"

        # =========================
        # LOAD PARAMS.YAML
        # =========================

        try:

            with open("params.yaml", "r", encoding="utf-8") as f:
                params = yaml.safe_load(f) or {}

            tokenizer_cfg = (
                params
                .get("preprocess_datasets", {})
                .get("tokenizer", {})
            )

        except FileNotFoundError:

            tokenizer_cfg = {}

        # =========================
        # PRIORITY:
        # 1. ARGUMENT
        # 2. params.yaml
        # 3. DEFAULT
        # =========================

        self.backend = (
            backend
            or tokenizer_cfg.get("backend")
            or default_backend
        )

        self.model_name = (
            model_name
            or tokenizer_cfg.get("model_name")
            or default_model_name
        )

        # =========================
        if self.backend == "hf":
            if AutoTokenizer is None:
                raise ImportError(
                    "Brakuje transformers. pip install transformers"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif self.backend == "tiktoken":
            if tiktoken is None:
                raise ImportError(
                    "Brakuje tiktoken. pip install tiktoken"
                )
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)

        else:
            raise ValueError(
                "backend musi być: 'hf' albo 'tiktoken'"
            )

    def tokenize(self, context: str) -> Dict[str, Any]:

        # =========================
        if self.backend == "hf":
            token_ids = self.tokenizer.encode(
                context,
                add_special_tokens=False
            )
            tokens = self.tokenizer.convert_ids_to_tokens(
                token_ids
            )
        elif self.backend == "tiktoken":
            token_ids = self.tokenizer.encode(context)
            # tiktoken nie ma normalnego convert_ids_to_tokens
            tokens = [
                self.tokenizer.decode([x])
                for x in token_ids
            ]

        return {
            "token_count": len(token_ids),
            "tokens": tokens,
            "token_ids": token_ids,
        }
    
    def get_tokenizer(self):
        return self.tokenizer