#!/usr/bin/env python
import argparse
import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

from src.chunking.chunking_text_one import chunk_text_one
from src.tools.logging_config import setup_logging
from src.tools.models_cache import clear_all_caches, get_tokenizer_service
from src.tools.presets import iter_cfg_with_presets

setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()


def chunking_text_all(
    datasets_cfg,
    chunking_cfg,
    splits,
    datasets_dir: Path,
    chunking_dir: Path,
):
    is_all = False
    load_dotenv()

    if splits == ["all"]:
        splits = ["train", "validation", "test"]
        is_all = True

    elif "all" in splits:
        is_all = True
        splits = [s for s in splits if s != "all"]
        for split in ["train", "validation", "test"]:
            if split not in splits:
                splits.append(split)
    for (
        (chunking_name, chunking_preset),
        (dataset_name, dataset_preset),
    ) in product(
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(datasets_cfg),
    ):
        clear_all_caches()
        tokenizer = get_tokenizer_service()
        dataset_preset_name = dataset_preset["name"]

        chunking_preset_name = chunking_preset["name"]
        chunking_preset_params = chunking_preset["params"]

        input_dir = datasets_dir / dataset_name / dataset_preset_name

        result_dir = chunking_dir / dataset_name / dataset_preset_name / chunking_name / chunking_preset_name

        non_existing_splits = [
            split for split in splits if not any(p.is_file() for p in (result_dir / split).glob("**/*"))
        ]

        if result_dir.exists() and not non_existing_splits:
            logger.info(f"⏭️ Pomijam {result_dir} – już istnieje")
            non_existing_splits = splits
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"▶ Chunking {result_dir}")

        chunk_text_one(
            chunking_type=chunking_name,
            params=chunking_preset_params,
            dataset_name=dataset_name,
            splits=non_existing_splits,
            input_dir=input_dir,
            output_dir=result_dir,
            tokenizer=tokenizer
        )
    if is_all:
        for (
            (chunking_name, chunking_preset),
            (dataset_name, dataset_preset),
        ) in product(
            iter_cfg_with_presets(chunking_cfg),
            iter_cfg_with_presets(datasets_cfg),
        ):
            dataset_preset_name = dataset_preset["name"]
            chunking_preset_name = chunking_preset["name"]

            result_dir = (
                chunking_dir
                / dataset_name
                / dataset_preset_name
                / chunking_name
                / chunking_preset_name
            )

            merge_books_and_meta(result_dir)
    clear_all_caches()



def merge_books_and_meta(result_dir: Path) -> None:
    """
    Tworzy:
        result_dir / all / Books
        result_dir / all / meta.json

    1. Kopiuje wszystkie jsonl z:
        result_dir/{train,test,validation}/Books

    2. Jeśli all/Books już istnieje i zawiera pliki -> przerywa.

    3. Nadaje globalnie unikalne chunk_id:
        - pierwszy plik: 0..N
        - drugi: N+1...
        itd.

    4. Łączy meta z:
        result_dir/{train,test,validation}/meta.json

        - split -> "all"
        - chunking_method i chunking_params bierze z train
        - sumuje:
            total_chunking_time
            total_wall_time
            chunks_num
    """

    splits = ["train", "test", "validation"]

    all_dir = result_dir / "all"
    all_books_dir = all_dir / "Books"

    # jeśli istnieje i są pliki -> skip
    if all_books_dir.exists() and any(all_books_dir.iterdir()):
        print(f"[SKIP] {all_books_dir} already contains files")
        return

    all_books_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # COPY + REMAP chunk_id
    # =========================

    global_chunk_id = 0

    for split in splits:
        books_dir = result_dir / split / "Books"

        if not books_dir.exists():
            print(f"[WARN] Missing: {books_dir}")
            continue

        jsonl_files: List[Path] = sorted(books_dir.glob("*.jsonl"))

        for src_file in jsonl_files:
            dst_file = all_books_dir / src_file.name

            with open(src_file, "r", encoding="utf-8") as f_in, open(
                dst_file,
                "w",
                encoding="utf-8",
            ) as f_out:

                for line in f_in:
                    row = json.loads(line)

                    row["chunk_id"] = global_chunk_id
                    global_chunk_id += 1

                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Total chunk_ids assigned: {global_chunk_id}")

    # =========================
    # MERGE META
    # =========================

    total_chunking_time = 0.0
    total_wall_time = 0.0
    chunks_num = 0

    train_meta = None

    for split in splits:
        meta_path = result_dir / split / "meta.json"

        if not meta_path.exists():
            print(f"[WARN] Missing meta: {meta_path}")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if split == "train":
            train_meta = meta

        total_chunking_time += meta.get("total_chunking_time", 0.0)
        total_wall_time += meta.get("total_wall_time", 0.0)
        chunks_num += meta.get("chunks_num", 0)

    if train_meta is None:
        raise ValueError("Missing train/meta.json")

    merged_meta = {
        "split": "all",
        "chunking_method": train_meta["chunking_method"],
        "chunking_params": train_meta["chunking_params"],
        "total_chunking_time": total_chunking_time,
        "total_wall_time": total_wall_time,
        "chunks_num": chunks_num,
    }

    meta_out_path = all_dir / "meta.json"

    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved merged meta -> {meta_out_path}")



def main():
    load_dotenv()
    DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
    CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR"))
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    chunking_cfg = params.get("chunking").get("methods")
    if not chunking_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")

    splits = params.get("chunking").get("splits")

    chunking_text_all(datasets_cfg, chunking_cfg, splits, DATASETS_DIR, CHUNKS_DIR)


if __name__ == "__main__":
    main()
