import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", "downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

repo_url = f"https://huggingface.co/datasets/NovelQA/NovelQA"

# folder docelowy
repo_path = DOWNLOADS_DIR / "NovelQA"

if not repo_path.exists():
    # wklejamy token bezpośrednio do URL
    subprocess.run([
        "git", "clone",
        f"https://user:{HF_TOKEN}@huggingface.co/datasets/NovelQA/NovelQA",
        str(repo_path)
    ], check=True)
else:
    print(f"Repozytorium już istnieje: {repo_path}")