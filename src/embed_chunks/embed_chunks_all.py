#!/usr/bin/env python
import argparse
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.embed_chunks.get_embedding import embed_chunks
from src.tools.presets import load_presets


def embed_chunks_all(datasets_cfg, chunking_cfg, embed_methods, chunks_dir: Path, vector_dir: Path):
    splits = ["train", "validation", "test"]
    for dataset_name, dataset_params in datasets_cfg.items():
        dataset_presets = load_presets(
            dataset_params["presets_file"],
            dataset_params.get("use_presets"),
        )
        for dataset_preset in dataset_presets:
            dataset_preset_name = dataset_preset["name"]
            
            for chunking_name, chunking_cfg in chunking_cfg.items():
                chunking_presets = load_presets(
                    chunking_cfg["presets_file"],
                    chunking_cfg.get("use_presets"),
                )
                for chunking_preset in chunking_presets:
                    chunking_preset_name = chunking_preset["name"]

                    for embed_name, embed_cfg in embed_methods.items():
                        embed_presets = load_presets(
                            embed_cfg["presets_file"],
                            embed_cfg.get("use_presets"),
                        )    
                        for embed_preset in embed_presets:
                            embed_preset_name = embed_preset["name"]
                            embed_preset_params = embed_preset["params"]
                            
                            splits = ["train", "validation", "test"]
                            for split in splits:
                                chunks_input_dir = chunks_dir / dataset_name / dataset_preset_name / chunking_name / chunking_preset_name / split
                                result_dir = vector_dir / dataset_name / dataset_preset_name / chunking_name / chunking_preset_name / embed_name / embed_preset_name / split

                                if not chunks_input_dir.exists():
                                    print(f"❌ Nie znaleziono folderu z chunkami: {chunks_input_dir}, pomijam...")
                                    continue

                                if result_dir.exists() and any(result_dir.iterdir()):
                                    print(f"⏭️ Pomijam {result_dir}, vector db już istnieje")
                                    continue
                                else:
                                    result_dir.mkdir(parents=True, exist_ok=True)
                                    print(f"➡️ Tworzę embeddingi: {result_dir}")
                                    embed_chunks(embed_name, embed_preset_params, chunks_input_dir, result_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=json.loads,
        nargs="+",
        required=True,
        help="Lista datasetów do przetworzenia",
    )
    parser.add_argument(
        "--chunking_types",
        type=json.loads,
        nargs="+",
        required=True,
        help="Typ chunkingu: fixed-size lub recursive",
    )
    parser.add_argument(
        "--embed_methods",
        type=json.loads,
        required=True,
        help="JSON string opisujący metody embeddowania i ich presety",
    )
    args = parser.parse_args()
    load_dotenv()

    CHUNKS_DIR = Path(os.getenv('CHUNKS_DIR'))
    VECTOR_DIR = Path(os.getenv('VECTOR_DIR'))
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    embed_chunks_all(
        datasets_cfg=args.datasets,
        chunking_cfg=args.chunking_types,
        embed_methods=args.embed_methods,
        chunks_dir=CHUNKS_DIR,
        vector_dir=VECTOR_DIR
    )


if __name__ == "__main__":
    main()
