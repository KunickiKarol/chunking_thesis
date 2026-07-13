import json
from pathlib import Path

import pandas as pd


def load_chunks_dataframes(
    chunks_path: Path,
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
        df = load_chunks_dataframe(
            chunks_path=chunks_path,
            config=config,
        )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_index()


def load_chunks_dataframe(
    chunks_path: Path,
    config: dict,
) -> pd.DataFrame:

    config_paths = [v for v in config.values() if v is not None]

    base_path = Path(chunks_path, *config_paths)
    chunks_results_dir = base_path / "Books"

    if not chunks_results_dir.exists():
        raise FileNotFoundError(f"Brak katalogu: {chunks_results_dir}")

    rows = []

    for jsonl_file in chunks_results_dir.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                rows.append(
                    {
                        **data,
                        **config,
                    }
                )

    df = pd.DataFrame(rows)
    index_cols = list(config.keys())
    df = df.set_index(index_cols).sort_index()


    return df