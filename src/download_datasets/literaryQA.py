import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv




def download_literaryQA():
    """
    Pobiera repozytorium LiteraryQA z GitHub i uruchamia skrypt pobierający książki.
    Jeśli repo już istnieje, pomija klonowanie i instalację zależności.
    """

    load_dotenv()
    DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR"))
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    repo_url = "https://github.com/KunickiKarol/LiteraryQA.git"
    repo_path = DOWNLOADS_DIR / "LiteraryQA"
    script_path = repo_path / "scripts" / "download_and_clean_books.py"

    repo_exists = repo_path.exists()

    # 1️⃣ Klonowanie tylko jeśli repo nie istnieje
    if not repo_exists:
        print(f"Klonuję repozytorium {repo_url} do {repo_path}...")
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True,
        )

        # 2️⃣ Instalacja zależności tylko przy świeżym klonie
        pyproject_file = repo_path / "pyproject.toml"
        if not pyproject_file.exists():
            raise FileNotFoundError(
                f"{pyproject_file} nie istnieje. Nie można zainstalować zależności."
            )

        print("Instaluję wymagane pakiety w aktualnym środowisku...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=repo_path,
            check=True,
        )
        cwd_path = repo_path
            # 3️⃣ Sprawdzenie skryptu
        if not script_path.exists():
            raise FileNotFoundError(f"Nie znaleziono skryptu: {script_path}")
    else:
        print(f"Repozytorium {repo_path} już istnieje – pomijam clone i pip install.")
        print(repo_path)
        cwd_path = repo_path
        #script_path = Path("scripts") / "download_and_clean_books.py"
        #output_dir = Path("data") / "script_pathliteraryqa"



    # 4️⃣ Uruchomienie skryptu
    script_path = Path("scripts") / "download_and_clean_books.py"
    output_dir = Path("data") / "literaryqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pobieram i czyszczę książki, zapisuję w {output_dir}...")
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--output_dir",
            str(output_dir),
        ],
        cwd=cwd_path,
        env=env,
        check=True,
    )

    print("Pobieranie LiteraryQA zakończone!")
    return repo_path, output_dir


if __name__ == "__main__":
    download_literaryQA()
