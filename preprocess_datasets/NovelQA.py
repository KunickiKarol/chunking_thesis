import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 1️⃣ Wczytaj zmienne środowiskowe z .env
load_dotenv()
DATASETS_DIR = Path(os.getenv("DATASETS", "datasets"))  # default = datasets

# 2️⃣ Foldery źródłowe
DOWNLOADS_DIR = Path("downloads/NovelQA")
BOOKS_SRC = DOWNLOADS_DIR / "Books"
DATA_SRC = DOWNLOADS_DIR / "Data"
BOOKMETA_SRC = DOWNLOADS_DIR / "bookmeta.json"

# 3️⃣ Foldery docelowe
BOOKS_DST = DATASETS_DIR / "NovelQA" / "Books"
TASKS_DST = DATASETS_DIR / "NovelQA" / "Tasks"
BOOKMETA_DST = DATASETS_DIR / "NovelQA"

# 4️⃣ Utwórz foldery docelowe jeśli nie istnieją
BOOKS_DST.mkdir(parents=True, exist_ok=True)
TASKS_DST.mkdir(parents=True, exist_ok=True)
BOOKMETA_DST.mkdir(parents=True, exist_ok=True)

# 5️⃣ Kopiowanie plików .txt z Books
for txt_file in BOOKS_SRC.rglob("*.txt"):  # rekurencyjnie wszystkie .txt
    dest_file = BOOKS_DST / txt_file.name
    shutil.copy2(txt_file, dest_file)
    print(f"Skopiowano {txt_file} → {dest_file}")

# 6️⃣ Kopiowanie plików .json z Data
for json_file in DATA_SRC.rglob("*.json"):
    dest_file = TASKS_DST / json_file.name
    shutil.copy2(json_file, dest_file)
    print(f"Skopiowano {json_file} → {dest_file}")

# 7️⃣ Kopiowanie bookmeta.json
if BOOKMETA_SRC.exists():
    dest_file = BOOKMETA_DST / "bookmeta.json"
    shutil.copy2(BOOKMETA_SRC, dest_file)
    print(f"Skopiowano {BOOKMETA_SRC} → {dest_file}")
else:
    print(f"Nie znaleziono {BOOKMETA_SRC}")
