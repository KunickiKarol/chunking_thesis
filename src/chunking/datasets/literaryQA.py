import json
import os
from pathlib import Path

from src.chunking.methods.chunking_fixed_size import chunking_fixed_size

from src.chunking.methods.chunking_recurisive import chunking_recursive


def chunk_text_literaryQA(chunking_type: str, dataset_name: str, result_dir: Path):
    """
    Przetwarza wszystkie pliki .txt w folderze datasetu i zapisuje chunki w output_folder.

    Args:
        dataset_name (str): nazwa datasetu (np. 'NovelQA')
        chunking_type (str): 'fixed-size' lub 'recursive'
        result_dir (Path): folder do zapisu chunków
    """
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    dataset_dir = DATASETS_DIR / dataset_name / "Books"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder datasetu nie istnieje: {dataset_dir}")

    splits = ["train", "test", "validation"]

    for split in splits:
        split_input_dir = dataset_dir / split
        split_output_dir = result_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(split_input_dir):
            if not filename.endswith(".txt"):
                continue

            path = split_input_dir / filename
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # Wybieramy metodę chunkowania
            if chunking_type == "fixed-size":
                chunks = chunking_fixed_size(text)
            elif chunking_type == "recursive":
                chunks = chunking_recursive(text)
            else:
                raise ValueError(f"Nieznany typ chunkingu: {chunking_type}")

            base_filename = filename.rsplit(".", 1)[0]
            output_file = split_output_dir / f"{base_filename}.jsonl"

            with open(output_file, "w", encoding="utf-8") as out_f:
                for i, chunk in enumerate(chunks):
                    record = {
                        "dataset": dataset_name,
                        "split": split,  # opcjonalne, ale bardzo przydatne
                        "chunking_method": chunking_type,
                        "source_file": base_filename,
                        "chunk_id": i,
                        "text": chunk,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Zakończono chunkowanie LiteraryQA z użyciem '{chunking_type}', {dataset_name}. Wyniki zapisano w: {result_dir}")

