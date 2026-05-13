#!/usr/bin/env python
import argparse
from itertools import product
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml

from src.rerank.rerank_one import rerank_one
from src.tools.presets import iter_cfg_with_presets


def rerank_all(datasets_cfg, chunking_cfg, splits, embed_cfg, search_cfg, rerank_cfg, 
               dataset_dir: Path, chunks_dir: Path, embed_dir: Path, search_dir: Path, rerank_dir: Path):
    print(f"➡️ Tworzę embeddingi: {embed_dir}")
    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (embed_name, embed_preset),
        (search_name, search_preset),
        (rerank_name, rerank_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(embed_cfg),
        iter_cfg_with_presets(search_cfg),
        iter_cfg_with_presets(rerank_cfg),
        splits,
    ):

        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        embed_preset_name = embed_preset["name"]
        search_preset_name = search_preset["name"]
        rerank_preset_name = rerank_preset["name"]

        task_type = dataset_preset["params"]['task_type']
        rerank_preset_params = rerank_preset["params"]


        task_input_dir = (
            dataset_dir
            / dataset_name
            / dataset_preset_name
            / 'Tasks'
            / task_type
            / split
        )

        chunks_input_dir = (
            chunks_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / split
            / "Books"
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

        search_input_dir = (
            search_dir
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

        result_dir = (
            rerank_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / embed_name
            / embed_preset_name
            / search_name
            / search_preset_name
            / rerank_name
            / rerank_preset_name
            / split
        )


        if not task_input_dir.exists():
            print(f"❌ Brak zadań: {task_input_dir}, pomijam...")
            continue

        if not chunks_input_dir.exists():
            print(f"❌ Brak chunków: {chunks_input_dir}, pomijam...")
            continue

        if not embed_input_dir.exists():
            print(f"❌ Brak embeddingów: {embed_input_dir}, pomijam...")
            continue
        print(type(search_input_dir), search_input_dir)
        if not search_input_dir.exists():
            print(f"❌ Brak search: {search_input_dir}, pomijam...")
            continue
        print(type(result_dir), result_dir)
        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, vector db już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        
        rerank_one(
            rerank_name,
            rerank_preset_params,
            split,
            task_input_dir,
            chunks_input_dir,
            embed_input_dir,
            search_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv('DATASETS_DIR'))
    CHUNKS_DIR = Path(os.getenv('CHUNKS_DIR'))
    EMBED_DIR = Path(os.getenv('EMBED_DIR'))
    SEARCH_DIR = Path(os.getenv('SEARCH_DIR'))
    RERANK_DIR = Path(os.getenv('RERANK_DIR'))
    RERANK_DIR.mkdir(parents=True, exist_ok=True)
    
    
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
        raise ValueError("Nie znaleziono vector_embed_methods w params.yaml")
    
    search_cfg = params.get("search").get("methods")
    if not search_cfg:
        raise ValueError("Nie znaleziono search_methods w params.yaml")
    
    rerank_cfg = params.get("rerank").get("methods")
    if not rerank_cfg:
        raise ValueError("Nie znaleziono rerank_methods w params.yaml")
    
    splits = params.get("chunking").get("splits")
    if not splits:
        raise ValueError("Nie znaleziono splits w params.yaml")
    
    rerank_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        splits=splits,
        embed_cfg=embed_cfg,
        search_cfg=search_cfg,
        rerank_cfg=rerank_cfg,
        dataset_dir=DATASET_DIR,
        chunks_dir=CHUNKS_DIR,
        embed_dir=EMBED_DIR,
        search_dir=SEARCH_DIR,
        rerank_dir=RERANK_DIR
    )


if __name__ == "__main__":
    main()
