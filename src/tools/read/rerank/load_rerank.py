import json
from pathlib import Path

import pandas as pd


def load_analyze_retrieval_dataframes(
    analyze_retrieval_path: Path,
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
        df = load_analyze_retrieval_dataframe(
            analyze_retrieval_path=analyze_retrieval_path,
            config=config,
        )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_index()


def load_analyze_retrieval_dataframe(
    analyze_retrieval_path: Path,
    config: dict,
) -> pd.DataFrame:

    config_paths = [
        value
        for value in config.values()
        if value is not None
    ]

    base_path = Path(
        analyze_retrieval_path,
        *config_paths,
    )

    analyze_retrieval_results_file = base_path / "analyze_retrieval_results.json"

    if not analyze_retrieval_results_file.exists():
        raise FileNotFoundError(
            f"Brak pliku analyze_retrieval_results.json: {analyze_retrieval_results_file}"
        )

    with open(analyze_retrieval_results_file, "r", encoding="utf-8") as f:
        analyze_retrieval_results = json.load(f)
    rows = {
        "rerank_result_time": analyze_retrieval_results["rerank_result_time"],
        **config,
    }

    for metric_key, metrics_val in analyze_retrieval_results["metrics"].items():

        # lista -> rozbij na osobne kolumny
        if isinstance(metrics_val, list):

            # np. Accuracy@1-5 -> Accuracy@1, Accuracy@2 ...
            if "-" in metric_key and "@" in metric_key:
                base_name, range_part = metric_key.split("@")
                start, end = map(int, range_part.split("-"))

                for idx, val in enumerate(metrics_val, start=start):
                    rows[f"{base_name}@{idx}"] = val

            else:
                for i, val in enumerate(metrics_val, start=1):
                    rows[f"{metric_key}_{i}"] = val

        # słownik -> rozbij po kluczach
        elif isinstance(metrics_val, dict):

            for sub_key, sub_val in metrics_val.items():
                rows[f"{metric_key}_{sub_key}"] = sub_val

        # zwykła wartość
        else:
            rows[metric_key] = metrics_val


    # dataframe z jednego rekordu
    df = pd.DataFrame([rows])

    index_cols = list(config.keys())

    df = df.set_index(index_cols).sort_index()

    return df