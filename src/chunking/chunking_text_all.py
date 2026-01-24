#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src import chunking
from src.chunking.datasets.literaryQA import chunk_text_literaryQA
from src.chunking.datasets.novelQA import chunk_text_novelQA
from src.tools.presets import load_presets

def chunking_text_all(datasets_cfg, chunking_cfg, datasets_dir: Path, chunking_dir: Path):
    for dataset_name, dataset_params in datasets_cfg.items():
        dataset_presets = load_presets(
            dataset_params["presets_file"],
            dataset_params.get("use_presets"),
        )
        for dataset_preset in dataset_presets:
            dataset_preset_name = dataset_preset["name"]
            dataset_preset_params = dataset_preset["params"]
            for chunking_name, chunking_params in chunking_cfg.items():
                chunking_presets = load_presets(
                    chunking_params["presets_file"],
                    chunking_params.get("use_presets"),
                )

                for chunking_preset in chunking_presets:
                    chunking_preset_name = chunking_preset["name"]
                    chunking_preset_params = chunking_preset["params"]
                    input_dir = (
                        datasets_dir
                        / dataset_name
                        / dataset_preset_name
                    )
                    result_dir = (
                        chunking_dir
                        / dataset_name
                        / dataset_preset_name
                        / chunking_name
                        / chunking_preset_name
                    )

                    if result_dir.exists() and any(result_dir.iterdir()):
                        print(f"⏭️ Pomijam {result_dir}")
                        continue

                    result_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"▶ Chunking {dataset_name} | {dataset_preset_name} | {chunking_name} | preset={chunking_preset_name}"
                    )

                    if dataset_name == "NovelQA":
                        chunk_text_novelQA(
                            chunking_type=chunking_name,
                            params=chunking_preset_params,
                            input_dir=input_dir,
                            output_dir=result_dir,
                        )

                    elif dataset_name == "LiteraryQA":
                        chunk_text_literaryQA(
                            chunking_type=chunking_name,
                            dataset_name=dataset_name,
                            output_dir=result_dir,
                            params=chunking_preset_params,
                        )

                    else:
                        raise NotImplementedError(
                            f"Nie zaimplementowano chunkingu dla {dataset_name}"
                        )


def main():
    load_dotenv()
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR"))
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=json.loads,
        required=True,
        help="Dict datasetów z params.yaml (JSON string)",
    )

    parser.add_argument(
        "--chunking_methods",
        type=json.loads,
        required=True,
        help="Dict metod chunkingu z params.yaml (JSON string)",
    )

    args = parser.parse_args()
    chunking_text_all(args.datasets, args.chunking_methods, DATASETS_DIR, CHUNKS_DIR)

if __name__ == "__main__":
    main()