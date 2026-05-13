import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.download_datasets.infiniteBench import download_infiniteBench
from src.download_datasets.literaryQA import download_literaryQA
from src.download_datasets.novelQA import download_novelQA


def download_datasets_all(datasets_download, downloads_dir: Path) -> None:
    """
    Pobiera dataset(y) do folderu downloads_dir.
    Pomija już pobrane dane.
    """
    for dataset in datasets_download:
        dataset_path = downloads_dir / dataset

        # Sprawdzenie czy folder istnieje i nie jest pusty
        if dataset_path.exists() and any(p.is_file() for p in dataset_path.iterdir()):
            print(f"⏭️ {dataset} już pobrany w {dataset_path}, pomijam.")
            continue

        dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"⬇️ Pobieram {dataset}...")

        # Wywołanie odpowiedniej funkcji pobierającej
        if dataset == "novelQA":
            download_novelQA()
        elif dataset == "literaryQA":
            download_literaryQA()
        elif dataset == "infiniteBench":
            download_infiniteBench()
        else:
            print(f"❌ Nie ma obsługi pobierania dla datasetu {dataset}")


def main() -> None:
    load_dotenv()
    DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", "data/downloads"))
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_download = params.get("datasets_download", [])
    if not datasets_download:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    download_datasets_all(datasets_download, DOWNLOADS_DIR)


if __name__ == "__main__":
    main()
