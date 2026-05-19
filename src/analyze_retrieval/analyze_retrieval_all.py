#!/usr/bin/env python
import os
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.analyze_retrieval.analyze_retrieval_one import analyze_retrieval_one
from src.tools.presets import iter_cfg_with_presets


def analyze_retrieval_all(
    datasets_cfg,
    chunking_cfg,
    splits,
    embed_cfg,
    search_cfg,
    rerank_cfg,
    analyze_retrieval_cfg,
    dataset_dir: Path,
    chunks_dir: Path,
    rerank_dir: Path,
    analyze_retrieval_dir: Path,
):
    print(f"➡️ Analyze Retrieval: {analyze_retrieval_dir}")
    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (embed_name, embed_preset),
        (search_name, search_preset),
        (rerank_name, rerank_preset),
        (analyze_retrieval_name, analyze_retrieval_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(embed_cfg),
        iter_cfg_with_presets(search_cfg),
        iter_cfg_with_presets(rerank_cfg),
        iter_cfg_with_presets(analyze_retrieval_cfg),
        splits,
    ):
    
        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        embed_preset_name = embed_preset["name"]
        search_preset_name = search_preset["name"]
        analyze_retrieval_preset_name = analyze_retrieval_preset["name"]
        rerank_preset_name = rerank_preset["name"]

        task_type = dataset_preset["params"]["task_type"]
        analyze_retrieval_preset_params = analyze_retrieval_preset["params"]
        if embed_preset["params"]["embed_type"] != 'global':
            print(f"⚠️ Pomijam {analyze_retrieval_name} {analyze_retrieval_preset_name}, bo embed_type != global ({embed_preset["params"]["embed_type"]})")
            continue
        task_input_dir = dataset_dir / dataset_name / dataset_preset_name / "Tasks" / task_type / split

        chunks_input_dir = (
            chunks_dir / dataset_name / dataset_preset_name / chunking_name / chunking_preset_name / split / "Books"
        )

        rerank_input_dir = (
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

        result_dir = (
            analyze_retrieval_dir
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
            / analyze_retrieval_name
            / analyze_retrieval_preset_name
            / split
        )

        if not task_input_dir.exists():
            print(f"❌ Brak zadań: {task_input_dir}, pomijam...")
            continue

        if not chunks_input_dir.exists():
            print(f"❌ Brak chunków: {chunks_input_dir}, pomijam...")
            continue

        if not rerank_input_dir.exists():
            print(f"❌ Brak rerank: {rerank_input_dir}, pomijam...")
            continue

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, generation już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)

        analyze_retrieval_one(
            analyze_retrieval_preset_params['analyze_type'],
            analyze_retrieval_preset_params,
            task_input_dir,
            chunks_input_dir,
            rerank_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv("DATASETS_DIR"))
    CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR"))
    RERANK_DIR = Path(os.getenv("RERANK_DIR"))
    ANALYZE_RETRIEVAL_DIR = Path(os.getenv("ANALYZE_RETRIEVAL_DIR"))
    ANALYZE_RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

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

    analyze_retrieval_cfg = params.get("analyze_retrieval").get("methods")
    if not analyze_retrieval_cfg:
        raise ValueError("Nie znaleziono analyze_retrieval_methods w params.yaml")

    splits = params.get("chunking").get("splits")
    if not splits:
        raise ValueError("Nie znaleziono splits w params.yaml")

    analyze_retrieval_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        splits=splits,
        embed_cfg=embed_cfg,
        search_cfg=search_cfg,
        rerank_cfg=rerank_cfg,
        analyze_retrieval_cfg=analyze_retrieval_cfg,
        dataset_dir=DATASET_DIR,
        chunks_dir=CHUNKS_DIR,
        rerank_dir=RERANK_DIR,
        analyze_retrieval_dir=ANALYZE_RETRIEVAL_DIR,
    )


if __name__ == "__main__":
    main()
