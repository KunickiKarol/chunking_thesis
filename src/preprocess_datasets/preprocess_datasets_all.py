import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml
import os


from ..tools.presets import iter_cfg_with_presets

# jawne importy datasetów
from .novelQA import NovelQAPreprocessor
from .literaryQA import LiteraryQAPreprocessor
from .infiniteBenchChoice import infiniteBenchChoicePreprocessor
from .infiniteBenchQA import infiniteBenchQAPreprocessor



PREPROCESSOR_MAP = {
    "novelQA": NovelQAPreprocessor,
    "literaryQA": LiteraryQAPreprocessor,
    "infiniteBenchQA": infiniteBenchQAPreprocessor,
    "infiniteBenchChoice": infiniteBenchChoicePreprocessor,
}


def preprocess_datasets_all(datasets_cfg, datasets_dir: Path):
    """
    datasets_cfg: dict z dataset -> {'presets_file': str, 'use_presets': list[str]}
    """
    for dataset_name, preset in iter_cfg_with_presets(datasets_cfg):

        try:
            PreprocessorCls = PREPROCESSOR_MAP[dataset_name]
        except KeyError:
            print(f"⚠️ Brak preprocessora dla {dataset_name}, pomijam")
            continue

        preset_name = preset["name"]
        preset_params = preset["params"]

        dataset_path = (
            datasets_dir
            / dataset_name
            / preset_name
        )

        if dataset_path.exists() and any(p.is_file() for p in dataset_path.iterdir()):
            print(f"⏭️ Pomijam {dataset_name}/{preset_name} – już istnieje")
            continue

        dataset_path.mkdir(parents=True, exist_ok=True)

        print(f"▶ Preprocessing {dataset_name} | preset={preset_name}")

        preprocessor = PreprocessorCls(
            dataset_path=dataset_path,
            params=preset_params,
        )
        preprocessor.run()


def main():
    load_dotenv()
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")


    preprocess_datasets_all(datasets_cfg, DATASETS_DIR)


if __name__ == "__main__":
    main()
