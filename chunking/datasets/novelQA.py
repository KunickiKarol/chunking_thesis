import os
import json
from pathlib import Path

from chunking.methods.chunking_fixed_size import chunking_fixed_size
from ..methods.chunking_recurisive import chunking_recursive


def chunk_text_novelQA(chunking_type: str, dataset_name: str, result_dir: Path):
    """
    Przetwarza wszystkie pliki .txt w folderze datasetu i zapisuje chunki w output_folder.

    Args:
        dataset_name (str): nazwa datasetu (np. 'NovelQA')
        chunking_type (str): 'fixed-size' lub 'recursive'
        result_dir (Path): folder do zapisu chunków
    """
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR", "datasets"))
    dataset_dir = DATASETS_DIR / dataset_name / "Books"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder datasetu nie istnieje: {dataset_dir}")

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".txt"):
            continue

        path = dataset_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Wybieramy metodę chunkowania
        if chunking_type == "fixed-size":
            chunks = chunking_fixed_size(text)
        elif chunking_type == "recursive":
            chunks = chunking_recursive(text)
        else:
            raise ValueError(f"Nieznany typ chunkingu: {chunking_type}")

        filename = filename.split('.')[0]
        output_file = result_dir / f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as out_f:
            for i, chunk in enumerate(chunks):
                record = {
                    "dataset": dataset_name,
                    "chunking_method": chunking_type,
                    "source_file": filename,
                    "chunk_id": i,
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ Zapisano {len(chunks)} chunków do {output_file}")
