from typing import List, Literal

import yaml
from chonkie import TokenChunker
from langchain_text_splitters import CharacterTextSplitter

from src.chunking.methods.register import register_chunker
from src.tools.tokenizer_service import TokenizerService


def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@register_chunker("fixed_size")
def chunking_fixed_size(
    text: str,
    chunk_size: int,
    overlap: int,
    mode: Literal["char", "token"] = "char",
    **kwargs,
) -> List[str]:

    params = load_params()
    tokenizer_cfg = params.get("preprocess_datasets").get("tokenizer")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # =========================
    # CHAR MODE (bez zmian)
    # =========================
    if mode == "char":
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    # =========================
    # TOKEN MODE (Chonkie + TokenizerService)
    # =========================
    elif mode == "token":

        backend = kwargs.get("backend") or tokenizer_cfg.get("backend")
        model_name = kwargs.get("model_name") or tokenizer_cfg.get("model_name")

        # 🔥 Twój service
        tokenizer_service = TokenizerService(
            backend=backend,
            model_name=model_name,
        )

        tokenizer = tokenizer_service.get_tokenizer()

        chunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        chunks = chunker.chunk(text)
        return [c.text for c in chunks]

    else:
        raise ValueError("mode must be 'char' or 'token'")
