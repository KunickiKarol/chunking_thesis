#!/usr/bin/env python
import argparse
from itertools import product
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml

from src.embed_chunks.get_embedding import embed_chunks
from src.tools.presets import iter_cfg_with_presets, load_presets


def embed_chunks_all(datasets_cfg, chunking_cfg, embed_cfg, chunks_dir: Path, embed_dir: Path):
    splits = ["train", "validation", "test"]

    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (embed_name, embed_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(embed_cfg),
        splits,
    ):
        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        embed_preset_name = embed_preset["name"]
        embed_preset_params = embed_preset["params"]

        chunks_input_dir = (
            chunks_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / split
        )

        result_dir = (
            embed_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / embed_name
            / embed_preset_name
            / split
        )

        if not chunks_input_dir.exists():
            print(f"❌ Brak chunków: {chunks_input_dir}, pomijam...")
            continue

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, vector db już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"➡️ Tworzę embeddingi: {result_dir}")
        embed_chunks(
            embed_name,
            embed_preset_params,
            chunks_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    CHUNKS_DIR = Path(os.getenv('CHUNKS_DIR'))
    EMBED_DIR = Path(os.getenv('EMBED_DIR'))
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    
    
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    chunking_cfg = params.get("chunking").get("methods")
    if not chunking_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")
    
    embed_cfg = params.get("vector_embed").get("methods")
    if not embed_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")
    
    embed_chunks_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        embed_cfg=embed_cfg,
        chunks_dir=CHUNKS_DIR,
        embed_dir=EMBED_DIR
    )


if __name__ == "__main__":
    main()
