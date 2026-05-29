import json
from pathlib import Path

import pandas as pd


def load_bookmeta_dataframe(
    data_dir: str,
    dataset_name: str,
    dataset_params_name: str,
) -> pd.DataFrame:

    bookmeta_path = Path(data_dir) / dataset_name / dataset_params_name / "bookmeta.json"

    if not bookmeta_path.exists():
        raise FileNotFoundError(f"Brak pliku: {bookmeta_path}")

    with open(bookmeta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for source, meta in data.items():
        row = dict(meta)
        row["source_file"] = source
        row["dataset_name"] = dataset_name
        row["dataset_params_name"] = dataset_params_name
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(["dataset_name", "dataset_params_name", "source_file"]).sort_index()

    return df


def load_multiple_bookmeta_dataframes(
    data_dir,
    dataset_names,
    dataset_params_names,
):

    all_dfs = []

    for dataset_name, dataset_params_name in zip(dataset_names, dataset_params_names):
        df = load_bookmeta_dataframe(
            data_dir=data_dir,
            dataset_name=dataset_name,
            dataset_params_name=dataset_params_name,
        )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_index()
