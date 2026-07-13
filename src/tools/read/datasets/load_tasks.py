import json
from pathlib import Path

import pandas as pd


def load_tasks_dataframe(
    data_dir: str,
    dataset_name: str,
    dataset_params_name: str,
    dataset_split_name: str
) -> pd.DataFrame:

    bookmeta_path = (
        Path(data_dir)
        / dataset_name
        / dataset_params_name
        / "Tasks"
        / "choice_qa"
        / dataset_split_name
    )

    if not bookmeta_path.exists():
            bookmeta_path = (
                Path(data_dir)
                / dataset_name
                / dataset_params_name
                / "Tasks"
                / "open_qa"
                / dataset_split_name
            )

    if not bookmeta_path.exists():
        raise FileNotFoundError(f"Brak katalogu: {bookmeta_path}")
    

    rows = []

    for json_file in bookmeta_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for question_id, question_content in data.items():
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_params_name": dataset_params_name,
                    "source_file": json_file.stem,
                    "question_id": question_id,
                    "split_name": dataset_split_name,
                    **question_content,
                }
            )

    df = pd.DataFrame(rows)

    return (
        df.set_index(
            ["dataset_name", "dataset_params_name"]
        )
        .sort_index()
    )


def load_tasks_dataframes(
    data_dir,
    tasks_configs
):
    
    all_dfs = []

    for config in tasks_configs:
        df = load_tasks_dataframe(
            data_dir=data_dir,
            dataset_name=config["dataset_name"],
            dataset_params_name=config["dataset_params_name"],
            dataset_split_name=config["split_name"]
        )
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_index()