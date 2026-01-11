import os
import shutil
from pathlib import Path
import subprocess
from dotenv import load_dotenv

load_dotenv()

def download(dataset_name: str, downloads_dir: str = None, hf_token: str = None):
    """
    Klonuje dataset z Hugging Face Hub do lokalnego folderu downloads.
    Jeśli folder już istnieje, usuwa go przed klonowaniem.

    Args:
        dataset_name (str): Nazwa datasetu na HF, np. "NovelQA/NovelQA"
        downloads_dir (str, optional): Ścieżka do folderu downloads. 
            Jeśli None, pobiera z .env lub domyślnie "downloads".
        hf_token (str, optional): Token Hugging Face. Jeśli None, pobiera z .env.
    """
    # 1️⃣ Wczytanie tokenu i folderu
    hf_token = hf_token or os.getenv("HF_TOKEN")
    downloads_dir = Path(downloads_dir or os.getenv("DOWNLOADS_DIR", "downloads"))
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Ścieżki
    repo_url = f"https://huggingface.co/datasets/{dataset_name}"
    repo_folder_name = dataset_name.split("/")[-1]  # np. "NovelQA"
    repo_path = downloads_dir / repo_folder_name

    # 3️⃣ Usuń istniejący folder
    if repo_path.exists():
        print(f"Usuwam istniejące repozytorium: {repo_path}")
        shutil.rmtree(repo_path)

    # 4️⃣ Klonowanie repozytorium z tokenem
    print(f"Klonuję repozytorium {repo_url} do {repo_path}...")
    subprocess.run([
        "git", "clone",
        f"https://user:{hf_token}@huggingface.co/datasets/{dataset_name}",
        str(repo_path)
    ], check=True)

    print("Klonowanie zakończone!")
    return repo_path


# 🔹 Przykład użycia
if __name__ == "__main__":
    clone_hf_dataset("NovelQA/NovelQA")
