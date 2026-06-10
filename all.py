from pathlib import Path
import shutil

BASE_DIR = Path("/workspace/chunking_thesis/data/datasets")

# znajdź wszystkie foldery DDD
# struktura: AAA/BBB/CCC/DDD
for ddd_dir in BASE_DIR.glob("*/*/*/*"):
    if not ddd_dir.is_dir():
        continue

    train_dir = ddd_dir / "train"
    test_dir = ddd_dir / "test"
    val_dir = ddd_dir / "validation"

    # interesują nas tylko katalogi które mają przynajmniej jeden z tych folderów
    source_dirs = [
        p for p in [train_dir, test_dir, val_dir]
        if p.exists() and p.is_dir()
    ]

    if not source_dirs:
        continue

    all_dir = ddd_dir / "all"
    all_dir.mkdir(exist_ok=True)

    print(f"\nProcessing: {ddd_dir}")

    for src_dir in source_dirs:
        print(f"  Copying from: {src_dir}")

        for file_path in src_dir.rglob("*"):
            if file_path.is_file():

                # zachowaj strukturę względem train/test/validation
                relative_path = file_path.relative_to(src_dir)

                destination = all_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)

                # jeśli istnieje konflikt nazw → dodaj prefix
                if destination.exists():
                    destination = (
                        destination.parent
                        / f"{src_dir.name}_{destination.name}"
                    )

                shutil.copy2(file_path, destination)

    print(f"  Done -> {all_dir}")

print("\nFinished.")