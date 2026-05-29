import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_infiniteBench():
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
    repo_url = f"https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench"
    repo_url_with_token = repo_url.replace("https://", f"https://user:{hf_token}@")
    repo_path = downloads_dir / "infiniteBench"

    # 3️⃣ Usuń istniejący folder
    if repo_path.exists():
        logger.info(f"Usuwam istniejące repozytorium: {repo_path}")
        shutil.rmtree(repo_path)

    # 4️⃣ Klonowanie repozytorium z tokenem
    logger.info(f"Klonuję repozytorium {repo_url} do {repo_path}...")
    snapshot_download(
        repo_id="xinrongzhang2022/InfiniteBench",
        repo_type="dataset",
        token=hf_token,
        local_dir=repo_path,
        allow_patterns=[
            ".gitattributes",
            "README.md",
            "longbook_choice_eng.jsonl",
            "longbook_qa_eng.jsonl",
        ],
    )
    logger.info("Klonowanie zakończone!")
    return repo_path


# 🔹 Przykład wywołania
if __name__ == "__main__":
    download_infiniteBench()
