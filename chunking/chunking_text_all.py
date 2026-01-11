#!/usr/bin/env python
import argparse
import yaml
from pathlib import Path

from .datasets.novelQA import chunk_text_novelQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Lista datasetów do przetworzenia",
    )
    parser.add_argument(
        "--chunking_types",
        type=str,
        nargs="+",
        required=True,
        help="Typ chunkingu: fixed-size lub recursive",
    )
    args = parser.parse_args()

    CHUNKING_DIR = Path("chunks")
    CHUNKING_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        for chunking_type in args.chunking_types:
            result_dir = CHUNKING_DIR / dataset_name / chunking_type
            if result_dir.exists():
                print(f"Pomijam {dataset_name}, chunki już istnieją")
                continue
            else:
                result_dir.mkdir(parents=True, exist_ok=True)
                
            if dataset_name == 'NovelQA':
                chunk_text_novelQA(chunking_type, dataset_name, result_dir)
            else:
                raise NotImplementedError(f'Nie zaimplementowano chunkowania dla {chunking_type}')


if __name__ == "__main__":
    main()
