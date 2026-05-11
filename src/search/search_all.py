#!/usr/bin/env python
import argparse
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml
from itertools import product


from src.embed_chunks.get_embedding import embed_chunks
from src.tools.presets import iter_cfg_with_presets


def search_all(datasets_cfg, chunking_cfg, embed_cfg,  search_cfg, dataset_dir: Path, embed_dir: Path, search_dir: Path):
    splits = ["train", "validation", "test"]

    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (embed_name, embed_preset),
        (search_name, search_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(embed_cfg),
        iter_cfg_with_presets(search_cfg),
        splits,
    ):
        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        embed_preset_name = embed_preset["name"]
        search_preset_name = search_preset["name"]

        task_type = dataset_preset["params"]['task_type']
        search_preset_params = search_preset["params"]

        task_input_dir = (
            dataset_dir
            / dataset_name
            / dataset_preset_name
            / 'Tasks'
            / task_type
            / split
        )
        
        embed_input_dir = (
            embed_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / embed_name
            / embed_preset_name
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
            / search_name
            / search_preset_name
            / split
        )

        if not task_input_dir.exists():
            print(f"❌ Brak zadań: {task_input_dir}, pomijam...")
            continue
        elif not embed_input_dir.exists():
            print(f"❌ Brak embeddingów: {embed_input_dir}, pomijam...")
            continue

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, vector db już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"➡️ Tworzę embeddingi: {result_dir}")
        search_query(
            search_name,
            search_preset_params,
            task_input_dir,
            embed_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv('DATASET_DIR'))
    EMBED_DIR = Path(os.getenv('EMBED_DIR'))
    SEARCH_DIR = Path(os.getenv('SEARCH_DIR'))
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    
    
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
    
    search_cfg = params.get("search").get("methods")
    if not search_cfg:
        raise ValueError("Nie znaleziono search_methods w params.yaml")

    search_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        embed_cfg=embed_cfg,
        search_cfg=search_cfg,
        dataset_dir=DATASET_DIR,
        embed_dir=EMBED_DIR,
        search_dir=SEARCH_DIR
    )


if __name__ == "__main__":
    main()
