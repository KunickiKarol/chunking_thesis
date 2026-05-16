#!/usr/bin/env python
import argparse
import json
import os
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.embed_chunks.get_embedding import embed_chunks
from src.search.search_query import search_query
from src.tools.presets import iter_cfg_with_presets


def analyze_embeddings_all(
    datasets_cfg, chunking_cfg, splits, embed_cfg, analyze_cfg, dataset_dir: Path, embed_dir: Path, analyze_embeddings_dir: Path
):
    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (embed_name, embed_preset),
        (analyze_name, analyze_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(embed_cfg),
        iter_cfg_with_presets(analyze_cfg),
        splits,
    ):
        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        embed_preset_name = embed_preset["name"]
        analyze_preset_name = analyze_preset["name"]

        embed_type = embed_preset["params"]["embed_type"]
        task_type = dataset_preset["params"]["task_type"]
        analyze_preset_params = analyze_preset["params"]

        task_input_dir = dataset_dir / dataset_name / dataset_preset_name / "Tasks" / task_type / split

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
            analyze_embeddings_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / embed_name
            / embed_preset_name
            / analyze_name
            / analyze_preset_name
            / split
        )

        if not task_input_dir.exists():
            print(f"❌ Brak zadań: {task_input_dir}, pomijam...")
            continue
        
        elif not embed_input_dir.exists():
            print(f"❌ Brak embeddingów: {embed_input_dir}, pomijam...")
            continue

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, analyze dir już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"➡️ Szukam query: {result_dir}")
        search_query(
            embed_type,
            analyze_preset_params,
            task_input_dir,
            embed_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv("DATASETS_DIR"))
    EMBED_DIR = Path(os.getenv("EMBED_DIR"))
    ANALYZE_EMBEDDINGS_DIR = Path(os.getenv("ANALYZE_EMBEDDINGS_DIR"))
    ANALYZE_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

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

    analyze_cfg = params.get("analyze_embeddings").get("methods")
    if not analyze_cfg:
        raise ValueError("Nie znaleziono analyze_embeddings_methods w params.yaml")

    splits = params.get("chunking").get("splits")
    if not splits:
        raise ValueError("Nie znaleziono splits w params.yaml")

    analyze_embeddings_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        splits=splits,
        embed_cfg=embed_cfg,
        analyze_cfg=analyze_cfg,
        dataset_dir=DATASET_DIR,
        embed_dir=EMBED_DIR,
        analyze_embeddings_dir=ANALYZE_EMBEDDINGS_DIR,
    )


if __name__ == "__main__":
    main()
