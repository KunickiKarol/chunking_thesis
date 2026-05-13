#!/usr/bin/env python
import argparse
import json
import os
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.chunking.chunking_text_one import chunk_text_one
from src.tools.presets import iter_cfg_with_presets


def chunking_text_all(
    datasets_cfg,
    chunking_cfg,
    splits,
    datasets_dir: Path,
    chunking_dir: Path,
):
    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
    ):
        dataset_preset_name = dataset_preset["name"]

        chunking_preset_name = chunking_preset["name"]
        chunking_preset_params = chunking_preset["params"]

        input_dir = datasets_dir / dataset_name / dataset_preset_name

        result_dir = chunking_dir / dataset_name / dataset_preset_name / chunking_name / chunking_preset_name

        if result_dir.exists() and any(p.is_file() for p in result_dir.glob("**/*")):
            print(f"⏭️ Pomijam {result_dir} – już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"▶ Chunking {dataset_name} | {dataset_preset_name} | " f"{chunking_name} | preset={chunking_preset_name}"
        )

        chunk_text_one(
            chunking_type=chunking_name,
            params=chunking_preset_params,
            dataset_name=dataset_name,
            splits=splits,
            input_dir=input_dir,
            output_dir=result_dir,
        )


def main():
    load_dotenv()
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR"))
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    chunking_cfg = params.get("chunking").get("methods")
    if not chunking_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")

    splits = params.get("chunking").get("splits")

    chunking_text_all(datasets_cfg, chunking_cfg, splits, DATASETS_DIR, CHUNKS_DIR)


if __name__ == "__main__":
    main()
