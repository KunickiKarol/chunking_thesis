import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.download_datasets.novelQA import download_novelQA
from src.download_datasets.literaryQA import download_literaryQA


def download_datasets_all(datasets_download, downloads_dir: Path) -> None:
    """
    Pobiera dataset(y) do folderu downloads_dir.
    Pomija już pobrane dane.
    """
    for dataset in datasets_download:
        dataset_path = downloads_dir / dataset

        # Sprawdzenie czy folder istnieje i nie jest pusty
        if dataset_path.exists() and any(dataset_path.iterdir()):
            print(f"⏭️ {dataset} już pobrany w {dataset_path}, pomijam.")
            continue

        dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"⬇️ Pobieram {dataset}...")

        # Wywołanie odpowiedniej funkcji pobierającej
        if dataset == "NovelQA":
            download_novelQA()
        elif dataset == "LiteraryQA":
            download_literaryQA()
        else:
            print(f"❌ Nie ma obsługi pobierania dla datasetu {dataset}")


def main() -> None:
    load_dotenv()
    DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", "data/downloads"))
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets_download",
        required=True,
        help="Lista datasetów do pobrania",
    )
    args = parser.parse_args()

    download_datasets_all(args.datasets_download, DOWNLOADS_DIR)


if __name__ == "__main__":
    main()
