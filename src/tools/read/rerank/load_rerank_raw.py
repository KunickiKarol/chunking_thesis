import json
from pathlib import Path

import pandas as pd


def load_retrieval_dataframes(
    retrieval_path: Path,
    configs: list[dict],
) -> pd.DataFrame:
    """
    configs = [
        {
            "dataset_name": "...",
            "dataset_params_name": "...",
            "chunking_name": "...",
            ...
        },
        ...
    ]
    """

    all_dfs = []

    for config in configs:
        df = load_retrieval_dataframe(
            retrieval_path=retrieval_path,
            config=config,
        )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_index()


def load_retrieval_dataframe(
    retrieval_path: Path,
    config: dict,
) -> pd.DataFrame:

    config_paths = [
        value
        for value in config.values()
        if value is not None
    ]

    base_path = Path(
        retrieval_path,
        *config_paths,
    )

    retrieval_results_file = base_path / "rerank_results.json"

    if not retrieval_results_file.exists():
        raise FileNotFoundError(
            f"Brak pliku rerank_results.json: {retrieval_results_file}"
        )

    with open(retrieval_results_file, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)
    rows = [
        {
            "question_id": key,
            "chunks_chosen": value,
            **config,
        }
        for key, value in retrieval_results.items()
    ]

    # dataframe z jednego rekordu
    df = pd.DataFrame(rows)

    index_cols = list(config.keys())

    df = df.set_index(index_cols).sort_index()

    return df