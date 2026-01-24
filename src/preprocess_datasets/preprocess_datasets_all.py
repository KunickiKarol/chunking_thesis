import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml
import os

from ..tools.presets import load_presets

# jawne importy datasetów
from .novelQA import NovelQAPreprocessor
from .literaryQA import LiteraryQAPreprocessor


def preprocess_datasets_all(datasets_cfg, datasets_dir: Path):
    """
    datasets_cfg: dict z dataset -> {'presets_file': str, 'use_presets': list[str]}
    """
    for dataset, cfg in datasets_cfg.items():
        dataset_path = datasets_dir / dataset
        
        presets = load_presets(cfg["presets_file"], cfg.get("use_presets"))

        for preset in presets:
            dataset_path = dataset_path / preset["name"]
            if dataset_path.exists() and any(dataset_path.iterdir()):
                print(f"⏭️ Pomijam {dataset}/{preset['name']} – już istnieje")
                continue
            dataset_path.mkdir(parents=True, exist_ok=True)

            print(f"Processing {dataset} with preset {preset['name']}...")
            if dataset.lower() == "novelqa":
                preprocessor = NovelQAPreprocessor(
                    dataset_path=dataset_path,
                    params=preset["params"]  # na razie tylko przechowywane
                )

                preprocessor.run()
            elif dataset.lower() == "literaryqa":
                preprocessor = LiteraryQAPreprocessor(
                    dataset_path=dataset_path,
                    params=preset["params"] # na razie tylko routing folderów
                )

                preprocessor.run()
            else:
                print(f"No preprocessing function for {dataset}")

def main():
    load_dotenv()
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets_cfg",
        type=json.loads,  # Python automatycznie parsuje JSON z args
        required=True,
        help="JSON mapping dataset -> presets_file + use_presets"
    )
    args = parser.parse_args()

    preprocess_datasets_all(args.datasets_cfg, DATASETS_DIR)

if __name__ == "__main__":
    main()
