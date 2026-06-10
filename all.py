from pathlib import Path
import shutil

BASE_DIR = Path("/workspace/chunking_thesis/data/datasets")

# Trzy wzorce ścieżek zawierających train/test/validation
patterns = [
    "*/*/*",       # AAA/BBB/Books  oraz  AAA/BBB/Tasks (głębokość 3)
    "*/*/*/**",    # AAA/BBB/Tags/CCC (głębokość 4+)
]

def merge_into_all(parent_dir: Path):
    """Kopiuje pliki z train/test/validation do folderu 'all' (pomija istniejące)."""
    train_dir = parent_dir / "train"
    test_dir  = parent_dir / "test"
    val_dir   = parent_dir / "validation"

    source_dirs = [
        p for p in [train_dir, test_dir, val_dir]
        if p.exists() and p.is_dir()
    ]
    if not source_dirs:
        return

    all_dir = parent_dir / "all"
    all_dir.mkdir(exist_ok=True)
    print(f"\nProcessing: {parent_dir}")

    copied = 0
    skipped = 0

    for src_dir in source_dirs:
        print(f"  Merging from: {src_dir.name}/")
        for file_path in src_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(src_dir)
            destination   = all_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists():
                skipped += 1
                continue  # pomiń – plik już istnieje w 'all'

            shutil.copy2(file_path, destination)
            copied += 1

    print(f"  Done -> {all_dir}  (skopiowano: {copied}, pominięto: {skipped})")


# --- Zbierz kandydatów ---
# Szukamy katalogów, które BEZPOŚREDNIO zawierają train/test/validation.
# Wystarczy przeszukać wszystkie podkatalogi BASE_DIR do pewnej głębokości.

candidates: set[Path] = set()

for depth_pattern in ["*/*/*", "*/*/*/*"]:
    for d in BASE_DIR.glob(depth_pattern):
        if not d.is_dir():
            continue
        # Sprawdź czy ten katalog ma przynajmniej jeden z trzech podfolderów
        if any((d / sub).is_dir() for sub in ("train", "test", "validation")):
            candidates.add(d)

for parent in sorted(candidates):
    merge_into_all(parent)

print("\nFinished.")