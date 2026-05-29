import logging
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def download_literaryQA():
    """
    Pobiera repozytorium LiteraryQA z GitHub,
    tworzy tymczasowe środowisko uv,
    instaluje zależności,
    uruchamia skrypt,
    a następnie usuwa środowisko.
    """

    load_dotenv()

    DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR"))
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    repo_url = "https://github.com/KunickiKarol/LiteraryQA.git"
    repo_path = DOWNLOADS_DIR / "literaryQA"

    script_path = repo_path / "scripts" / "download_and_clean_books.py"

    # Tymczasowy venv dla repo
    temp_venv = repo_path / ".tmp_venv"

    # Usuń stare repo
    if repo_path.exists():
        logger.info(f"Usuwam istniejące repozytorium: {repo_path}")
        shutil.rmtree(repo_path)

    # Klonowanie repo
    logger.info(f"Klonuję repozytorium {repo_url}...")
    subprocess.run(
        ["git", "clone", repo_url, str(repo_path)],
        check=True,
    )

    pyproject_file = repo_path / "pyproject.toml"

    if not pyproject_file.exists():
        raise FileNotFoundError(f"Nie znaleziono {pyproject_file}")

    # Tworzenie tymczasowego środowiska
    logger.info("Tworzę tymczasowe środowisko uv...")

    subprocess.run(
        ["uv", "venv", str(temp_venv), "--clear"],
        check=True,
        cwd=repo_path,
    )

    # Python z tymczasowego venv
    if os.name == "nt":
        python_bin = temp_venv / "Scripts" / "python.exe"
    else:
        python_bin = temp_venv / "bin" / "python"

    # Instalacja dependencies z pyproject.toml
    logger.info("Instaluję zależności...")

    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "-e",
            ".",
        ],
        cwd=repo_path,
        check=True,
    )

    # Output
    output_dir = repo_path / "data" / "literaryqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Uruchomienie skryptu
    logger.info("Uruchamiam download_and_clean_books.py...")
    script_path = Path("scripts") / "download_and_clean_books.py"

    subprocess.run(
        [
            str(python_bin),
            str(script_path),
            "--output_dir",
            "data/literaryqa",
        ],
        cwd=repo_path,
        env=env,
        check=True,
    )

    # Cleanup
    logger.info("Usuwam tymczasowe środowisko...")

    shutil.rmtree(temp_venv, ignore_errors=True)

    logger.info("Pobieranie LiteraryQA zakończone!")

    return repo_path, output_dir


if __name__ == "__main__":
    download_literaryQA()
