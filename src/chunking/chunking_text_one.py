import json
import time
from pathlib import Path
from typing import List

from src.chunking.methods.all_chunker import chunk_text


def chunk_text_one(
    chunking_type: str,
    params: dict,
    dataset_name: str,
    splits: List,
    input_dir: Path,
    output_dir: Path,
):
    """
    Chunkuje dataset_name na podstawie: input_dir
    """

    books_dir = input_dir / "Books"
    bookmeta_file = input_dir / "bookmeta.json"

    if not bookmeta_file.exists():
        raise FileNotFoundError(f"Missing bookmeta.json: {bookmeta_file}")

    with bookmeta_file.open("r", encoding="utf-8") as f:
        bookmeta = json.load(f)

    # tworzenie folderów splitów
    for split in splits:
        (output_dir / split / "Books").mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    chunks_train_num = 0
    chunks_test_num = 0
    chunks_validation_num = 0

    split_times = {split: 0.0 for split in splits}
    chunk_id = 0
    for split in splits:
        split_out_dir = output_dir / split
        split_books_dir = split_out_dir / "Books"

        split_chunks = 0
        start_split = time.perf_counter()

        is_example = split in ("example", "examples")

        for txt_file in books_dir.rglob("*.txt"):
            bookid = txt_file.stem
            meta = bookmeta.get(bookid)

            if is_example:
                if txt_file.parent.name != split or not meta.get(split):
                    continue
            else:
                if meta.get("split") != split:
                    continue

            with txt_file.open("r", encoding="utf-8") as f:
                text = f.read()
            start = time.perf_counter()
            chunks = chunk_text(chunking_type, text, **params)
            end = time.perf_counter()
            chunking_time = end - start
            split_times[split] += chunking_time

            out_file = split_books_dir / f"{bookid}.jsonl"
            with out_file.open("w", encoding="utf-8") as out:
                for chunk in chunks:
                    record = {
                        "source_file": bookid,
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "split": split,
                        "chunking_method": chunking_type,
                        "chunking_params": params,
                        "chunking_time": chunking_time,
                    }
                    chunk_id += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

            chunks_count = len(chunks)
            split_chunks += chunks_count
            total_chunks += chunks_count

            if split == "train":
                chunks_train_num += chunks_count
            elif split == "test":
                chunks_test_num += chunks_count
            elif split == "validation":
                chunks_validation_num += chunks_count

        end_split = time.perf_counter()

        # zapis meta dla splitu
        meta = {
            "split": split,
            "chunking_method": chunking_type,
            "chunking_params": params,
            "total_chunking_time": split_times[split],
            "total_wall_time": end_split - start_split,
            "chunks_num": split_chunks,
        }

        with open(output_dir / split / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ Split: {split} | time: {split_times[split]:.4f}s | chunks: {split_chunks}")

    print(
        f"\n🔥 DATASET SUMMARY {dataset_name}\n"
        f"Total chunks: {total_chunks}\n"
        f"Train: {chunks_train_num}, Test: {chunks_test_num}, Val: {chunks_validation_num}"
    )
