import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 1️⃣ Wczytaj zmienne środowiskowe z .env
load_dotenv()
  # default = datasets

def preprocess_novelQA():
    """
    Przetwarza pobrany dataset NovelQA:
    - Kopiuje wszystkie pliki .txt z Books
    - Kopiuje wszystkie pliki .json z Data
    - Kopiuje bookmeta.json

    Args:
        downloads_dir (str): folder źródłowy pobranego repozytorium NovelQA
        datasets_dir (Path): folder docelowy dla datasetów
    """

    DOWNLOADS_DIR =  Path(os.getenv("DOWNLOADS_DIR"))
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    
    BOOKS_SRC = DOWNLOADS_DIR / "NovelQA" / "Books"
    DATA_SRC = DOWNLOADS_DIR / "NovelQA" / "Data"
    BOOKMETA_SRC = DOWNLOADS_DIR / "NovelQA" / "bookmeta.json"

    # Foldery docelowe
    BOOKS_DST = DATASETS_DIR / "NovelQA" / "Books"
    TASKS_DST = DATASETS_DIR / "NovelQA" / "Tasks"
    BOOKMETA_DST = DATASETS_DIR / "NovelQA"

    # Utwórz foldery docelowe jeśli nie istnieją
    BOOKS_DST.mkdir(parents=True, exist_ok=True)
    TASKS_DST.mkdir(parents=True, exist_ok=True)
    BOOKMETA_DST.mkdir(parents=True, exist_ok=True)

    # Kopiowanie plików .txt z Books
    for txt_file in BOOKS_SRC.rglob("*.txt"):
        dest_file = BOOKS_DST / txt_file.name
        shutil.copy2(txt_file, dest_file)
        print(f"Skopiowano {txt_file} → {dest_file}")

    # Kopiowanie plików .json z Data
    for json_file in DATA_SRC.rglob("*.json"):
        dest_file = TASKS_DST / json_file.name
        shutil.copy2(json_file, dest_file)
        print(f"Skopiowano {json_file} → {dest_file}")

    # Kopiowanie bookmeta.json
    if BOOKMETA_SRC.exists():
        dest_file = BOOKMETA_DST / "bookmeta.json"
        shutil.copy2(BOOKMETA_SRC, dest_file)
        print(f"Skopiowano {BOOKMETA_SRC} → {dest_file}")
    else:
        print(f"Nie znaleziono {BOOKMETA_SRC}")

    print("Preprocessing NovelQA zakończony!")
    return BOOKS_DST, TASKS_DST, BOOKMETA_DST

# 🔹 Przykład wywołania
if __name__ == "__main__":
    preprocess_novelQA()
