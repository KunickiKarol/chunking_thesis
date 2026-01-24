import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def download_novelQA():
    """
    Klonuje dataset z Hugging Face Hub do lokalnego folderu downloads.
    Jeśli folder już istnieje, usuwa go przed klonowaniem.

    Args:
        dataset_name (str): Nazwa datasetu na HF, np. "NovelQA/NovelQA"
        downloads_dir (str, optional): Ścieżka do folderu downloads.
            Jeśli None, pobiera z .env lub domyślnie "downloads".
        hf_token (str, optional): Token Hugging Face. Jeśli None, pobiera z .env.
    """
    load_dotenv()
    # 1️⃣ Wczytanie tokenu i folderu
    hf_token = os.getenv("HF_TOKEN")
    downloads_dir = Path(os.getenv("DOWNLOADS_DIR"))
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Ścieżki
    repo_url = f"https://huggingface.co/datasets/NovelQA"
    repo_path = downloads_dir / "NovelQA"

    # 3️⃣ Usuń istniejący folder
    if repo_path.exists():
        print(f"Usuwam istniejące repozytorium: {repo_path}")
        shutil.rmtree(repo_path)

    # 4️⃣ Klonowanie repozytorium z tokenem
    print(f"Klonuję repozytorium {repo_url} do {repo_path}...")
    subprocess.run(
        [
            "git",
            "clone",
            f"https://user:{hf_token}@huggingface.co/datasets/NovelQA/NovelQA",
            str(repo_path),
        ],
        check=True,
    )

    print("Klonowanie zakończone!")
    return repo_path


# 🔹 Przykład wywołania
if __name__ == "__main__":
    download_novelQA()
