import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# jawne importy datasetów
from download_scripts.NovelQA import download as download_NovelQA

# 1️⃣ Wczytanie zmiennych środowiskowych z .env
load_dotenv()
DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 2️⃣ Parsowanie argumentów z wiersza poleceń
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", required=True)
args = parser.parse_args()

# 3️⃣ Iteracja po datasetach
for dataset in args.datasets:
    dataset_path = DOWNLOADS_DIR / dataset
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"{dataset} już pobrany w {dataset_path}, pomijam.")
        continue

    print(f"Pobieram {dataset}...")

    # 4️⃣ Wywołanie jawnie importowanej funkcji
    if dataset == "NovelQA":
        download_NovelQA(DOWNLOADS_DIR)
    else:
        print(f"Nie ma obsługi pobierania dla datasetu {dataset}")
