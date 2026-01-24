import json
from pathlib import Path

from src.chunking.methods.chunking_fixed_size import chunking_fixed_size
from ..methods.chunking_recurisive import chunking_recursive


def chunk_text_novelQA(
    chunking_type: str,
    params: dict,
    input_dir: Path,
    output_dir: Path,
):
    """
    Chunkuje NovelQA na podstawie:
      input_dir/
        ├── Books/
        │   └── train|validation|test/*.txt
        └── bookmeta.json

    Wynik:
      output_dir/
        └── train|validation|test/*.jsonl
    """

    books_dir = input_dir / "Books"
    bookmeta_file = input_dir / "bookmeta.json"

    if not bookmeta_file.exists():
        raise FileNotFoundError(f"Missing bookmeta.json: {bookmeta_file}")

    with bookmeta_file.open("r", encoding="utf-8") as f:
        bookmeta = json.load(f)

    # foldery splitów
    for split in ("train", "validation", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # iterujemy po książkach
    for txt_file in books_dir.rglob("*.txt"):
        bookid = txt_file.stem

        split = bookmeta.get(bookid, {}).get("split")
        split_out_dir = output_dir / split

        with txt_file.open("r", encoding="utf-8") as f:
            text = f.read()

        # --- CHUNKING ---
        if chunking_type == "fixed_size":
            chunks = chunking_fixed_size(text, **params)
        elif chunking_type == "recursive":
            chunks = chunking_recursive(text, **params)
        else:
            raise ValueError(f"Unknown chunking type: {chunking_type}")

        # --- SAVE ---
        out_file = split_out_dir / f"{bookid}.jsonl"
        with out_file.open("w", encoding="utf-8") as out:
            for i, chunk in enumerate(chunks):
                record = {
                    "source_file": bookid,
                    "chunk_id": i,
                    "text": chunk,
                    "split": split,
                    "chunking_method": chunking_type,
                    "chunking_params": params,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ NovelQA chunked → {output_dir}")
