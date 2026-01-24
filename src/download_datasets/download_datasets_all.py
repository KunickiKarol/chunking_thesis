import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.download_datasets.novelQA import download_novelQA
from src.download_datasets.literaryQA import download_literaryQA


def download_datasets_all(datasets_download, downloads_dir) -> None:
    # 3️⃣ Iteracja po datasetach
    for dataset in datasets_download:
        dataset_path = downloads_dir / dataset
        print(dataset_path, dataset_path.exists(), dataset_path.iterdir())
        if dataset_path.exists() and any(dataset_path.iterdir()):
            print(f"{dataset} już pobrany w {dataset_path}, pomijam.")
            continue

        print(f"Pobieram {dataset}...")

        # 4️⃣ Wywołanie jawnie importowanej funkcji
        if dataset == "NovelQA":
            download_novelQA()
        elif dataset == "LiteraryQA":
            download_literaryQA()
        else:
            print(f"Nie ma obsługi pobierania dla datasetu {dataset}")
            
def main() -> None:
    # 1️⃣ Wczytanie zmiennych środowiskowych z .env
    load_dotenv()
    DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR"))
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Parsowanie argumentów z wiersza poleceń
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_download", nargs="+", required=True)
    args = parser.parse_args()
    download_datasets_all(args.datasets_download, DOWNLOADS_DIR)


if __name__ == "__main__":
    main()