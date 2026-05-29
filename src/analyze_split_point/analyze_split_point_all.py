#!/usr/bin/env python
import logging
import os
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.analyze_split_point.analyze_split_point_one import analyze_split_point_one
from src.search.search_query import search_query
from src.tools.logging_config import setup_logging
from src.tools.presets import iter_cfg_with_presets

setup_logging()
logger = logging.getLogger(__name__)


def analyze_split_points_all(
    datasets_cfg, chunking_cfg, splits, analyze_cfg, dataset_dir: Path, chunks_dir: Path, analyze_split_points_dir: Path
):
    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (analyze_name, analyze_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(analyze_cfg),
        splits,
    ):
        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        analyze_preset_name = analyze_preset["name"]
        analyze_preset_name = analyze_preset["name"]

        task_type = dataset_preset["params"]["task_type"]
        analyze_type = analyze_preset["params"]["analyze_type"]

        analyze_preset_params = analyze_preset["params"]

        tags_input_dir = dataset_dir / dataset_name / dataset_preset_name / "Tags" / task_type / split
        books_input_dir = dataset_dir / dataset_name / dataset_preset_name / "Books" / split

        chunks_input_dir = (
            chunks_dir / dataset_name / dataset_preset_name / chunking_name / chunking_preset_name / split / "Books"
        )

        result_dir = (
            analyze_split_points_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / analyze_name
            / analyze_preset_name
            / split
        )

        if not tags_input_dir.exists():
            logger.info(f"❌ Brak tagów: {tags_input_dir}, pomijam...")
            continue

        elif not chunks_input_dir.exists():
            logger.info(f"❌ Brak chunków: {chunks_input_dir}, pomijam...")
            continue

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            logger.info(f"⏭️ Pomijam {result_dir}, analyze dir już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"➡️ Analizuję split points w {result_dir}")
        analyze_split_point_one(
            analyze_type,
            analyze_preset_params,
            chunks_input_dir,
            books_input_dir,
            tags_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv("DATASETS_DIR"))
    CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR"))
    ANALYZE_SPLIT_POINTS_DIR = Path(os.getenv("ANALYZE_SPLIT_POINTS_DIR"))
    ANALYZE_SPLIT_POINTS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    chunking_cfg = params.get("chunking").get("methods")
    if not chunking_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")

    analyze_cfg = params.get("analyze_split_points").get("methods")
    if not analyze_cfg:
        raise ValueError("Nie znaleziono analyze_split_points_methods w params.yaml")

    splits = params.get("chunking").get("splits")
    if not splits:
        raise ValueError("Nie znaleziono splits w params.yaml")

    analyze_split_points_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        splits=splits,
        analyze_cfg=analyze_cfg,
        dataset_dir=DATASET_DIR,
        chunks_dir=CHUNKS_DIR,
        analyze_split_points_dir=ANALYZE_SPLIT_POINTS_DIR,
    )


if __name__ == "__main__":
    main()
