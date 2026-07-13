import os
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _process_file(json_path: Path, root_dir: Path, depth: int):
    try:
        rel = json_path.relative_to(root_dir)
        parts = rel.parts

        if len(parts) != depth + 1:
            return None

        key = parts[:depth]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # szybciej: list comprehension bez if w środku
        verdicts = [
            v.get("verdict")
            for v in data.values()
            if "verdict" in v
        ]

        if not verdicts:
            return None

        correct = verdicts.count("correct")
        ratio = correct * 100 / len(verdicts)

        return (*key, ratio)

    except Exception:
        return None


def load_evaluation_results(root_dir: str) -> pd.DataFrame:
    index_names = [
        "dataset_name",
        "dataset_params_name",
        "chunking_name",
        "chunking_params_name",
        "embed_name",
        "embed_params_name",
        "search_name",
        "search_params_name",
        "rerank_name",
        "rerank_params_name",
        "generator_name",
        "generator_params_name",
        "judge_name",
        "judge_params_name",
        "split_name",
    ]

    root = Path(root_dir)
    depth = len(index_names)

    json_files = [
        p for p in root.rglob("generation_results.json")
        if len(p.relative_to(root).parts) == depth + 1
    ]

    if not json_files:
        raise ValueError(f"Nie znaleziono plików w: {root_dir}")

    records = []

    # ThreadPool → idealne dla I/O (czytanie plików)
    max_workers = min(32, (os.cpu_count() or 4) * 4)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_process_file, p, root, depth)
            for p in json_files
        ]

        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                records.append(res)

    df = pd.DataFrame(records, columns=index_names + ["answers_ratio"])
    return df.set_index(index_names).sort_index()


import os
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _process_file_raw(json_path: Path, root_dir: Path, depth: int):
    try:
        rel = json_path.relative_to(root_dir)
        parts = rel.parts

        if len(parts) != depth + 1:
            return None

        key = parts[:depth]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []

        for question, item in data.items():
            verdict = item.get("verdict")
            if verdict is None:
                continue

            rows.append((*key, question, 1 if verdict == "correct" else 0))

        return rows

    except Exception:
        return None


def load_evaluation_results_raw(root_dir: str) -> pd.DataFrame:
    index_names = [
        "dataset_name",
        "dataset_params_name",
        "chunking_name",
        "chunking_params_name",
        "embed_name",
        "embed_params_name",
        "search_name",
        "search_params_name",
        "rerank_name",
        "rerank_params_name",
        "generator_name",
        "generator_params_name",
        "judge_name",
        "judge_params_name",
        "split_name",
    ]

    root = Path(root_dir)
    depth = len(index_names)

    json_files = [
        p for p in root.rglob("generation_results.json")
        if len(p.relative_to(root).parts) == depth + 1
    ]

    if not json_files:
        raise ValueError(f"Nie znaleziono plików w: {root_dir}")

    records = []

    max_workers = min(32, (os.cpu_count() or 4) * 4)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_process_file_raw, p, root, depth)
            for p in json_files
        ]

        for fut in as_completed(futures):
            res = fut.result()
            if res:
                records.extend(res)

    df = pd.DataFrame(
        records,
        columns=index_names + ["question", "value"]
    )

    return df.set_index(index_names).sort_index()