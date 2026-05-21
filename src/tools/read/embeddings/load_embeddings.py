from pathlib import Path
import json
import numpy as np
import pandas as pd
from itertools import product


def load_multiple_embeddings_dataframes(
    embed_dir,
    dataset_names,
    dataset_params_names,
    chunking_names,
    chunking_params_names,
    embed_names,
    embed_params_names,
    split_names,
):

    all_dfs = []

    combinations = product(
        dataset_names,
        dataset_params_names,
        chunking_names,
        chunking_params_names,
        embed_names,
        embed_params_names,
        split_names,
    )

    for (
        dataset_name,
        dataset_params_name,
        chunking_name,
        chunking_params_name,
        embed_name,
        embed_params_name,
        split_name,
    ) in combinations:

        df = load_embeddings_dataframe(
            embed_dir=embed_dir,
            dataset_name=dataset_name,
            dataset_params_name=dataset_params_name,
            chunking_name=chunking_name,
            chunking_params_name=chunking_params_name,
            embed_name=embed_name,
            embed_params_name=embed_params_name,
            split_name=split_name,
        )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_index()


def load_embeddings_dataframe(
    embed_dir: str,
    dataset_name: str,
    dataset_params_name: str,
    chunking_name: str,
    chunking_params_name: str,
    embed_name: str,
    embed_params_name: str,
    split_name: str,
) -> pd.DataFrame:

    base_path = (
        Path(embed_dir)
        / dataset_name
        / dataset_params_name
        / chunking_name
        / chunking_params_name
        / embed_name
        / embed_params_name
        / split_name
    )

    meta_path = base_path / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Brak pliku meta.json: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    embed_type = meta.get("embed_type", "local")

    metadata_dir = base_path / "Metadatas"
    embeddings_dir = base_path / "Embeddings"

    rows = []

    # =========================================================
    # GLOBAL
    # =========================================================
    if embed_type == "global":

        metadata_file = metadata_dir / "global.json"
        embedding_file = embeddings_dir / "global.npy"

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        embeddings = np.load(embedding_file)

        for item in metadata_list:

            emb_id = item["id"]

            row = dict(item)

            row["dataset_name"] = dataset_name
            row["dataset_params_name"] = dataset_params_name
            row["chunking_name"] = chunking_name
            row["chunking_params_name"] = chunking_params_name
            row["embed_name"] = embed_name
            row["embed_params_name"] = embed_params_name
            row["split_name"] = split_name

            row["embedding"] = embeddings[emb_id]

            rows.append(row)

    # =========================================================
    # LOCAL
    # =========================================================
    else:

        metadata_files = list(metadata_dir.glob("*.json"))

        for metadata_file in metadata_files:

            source = metadata_file.stem
            embedding_file = embeddings_dir / f"{source}.npy"

            if not embedding_file.exists():
                continue

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)

            embeddings = np.load(embedding_file)

            for item in metadata_list:

                emb_id = item["id"]

                row = dict(item)

                row["dataset_name"] = dataset_name
                row["dataset_params_name"] = dataset_params_name
                row["chunking_name"] = chunking_name
                row["chunking_params_name"] = chunking_params_name
                row["embed_name"] = embed_name
                row["embed_params_name"] = embed_params_name
                row["split_name"] = split_name

                row["embedding"] = embeddings[emb_id]

                rows.append(row)

    df = pd.DataFrame(rows)

    index_columns = [
        "dataset_name",
        "dataset_params_name",
        "chunking_name",
        "chunking_params_name",
        "embed_name",
        "embed_params_name",
        "split_name",
        "source_file",
        "chunk_id",
    ]

    df = df.set_index(index_columns).sort_index()

    return df