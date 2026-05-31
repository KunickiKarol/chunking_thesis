import os
import json
import pandas as pd
from pathlib import Path


def load_evaluation_results(root_dir: str) -> pd.DataFrame:
    """
    Wczytuje wyniki ewaluacji z plików generation_results.json
    i zwraca DataFrame z MultiIndex.

    Poziomy MultiIndex (w kolejności):
        dataset_name, dataset_params_name,
        chunking_name, chunking_params_name,
        embed_name, embed_params_name,
        search_name, search_params_name,
        rerank_name, rerank_params_name,
        generator_name, generator_params_name,
        judge_name, judge_params_name,
        split_name

    Kolumna:
        answers_ratio  –  % verdictów == 'correct' spośród wszystkich
    """
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
    DEPTH = len(index_names)  # 15 poziomów

    records = []

    for json_path in Path(root_dir).rglob("generation_results.json"):
        # Relatywna ścieżka względem root_dir
        rel = json_path.relative_to(root_dir)
        parts = rel.parts  # ostatni element to "generation_results.json"

        if len(parts) != DEPTH + 1:
            print(f"Pomijam (nieoczekiwana głębokość {len(parts)-1}): {rel}")
            continue

        key = parts[:DEPTH]  # 15 segmentów ścieżki

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        verdicts = [v["verdict"] for v in data.values() if "verdict" in v]
        if not verdicts:
            continue

        correct = sum(1 for v in verdicts if v == "correct")
        ratio = correct / len(verdicts) * 100

        records.append((*key, ratio))

    if not records:
        raise ValueError(f"Nie znaleziono żadnych plików generation_results.json w: {root_dir}")

    columns = index_names + ["answers_ratio"]
    df = pd.DataFrame(records, columns=columns)
    df = df.set_index(index_names).sort_index()

    return df
