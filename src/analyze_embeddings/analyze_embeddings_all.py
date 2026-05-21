#!/usr/bin/env python
import os
from itertools import product
from pathlib import Path
import time

import yaml
from dotenv import load_dotenv

from src.analyze_embeddings import analyze_embeddings_one
from src.analyze_embeddings.methods.all_analyze_embeddings import get_analyze_embeddings
from src.search.search_query import search_query
from src.tools.presets import iter_cfg_with_presets, get_list_of_presets
from src.tools.read.datasets.load_metadata import load_multiple_bookmeta_dataframes
from src.tools.read.embeddings.load_embeddings import load_embeddings_dataframe, load_multiple_embeddings_dataframes


def analyze_embeddings_all(
    datasets_cfg, chunking_cfg, splits, embed_cfg, analyze_cfg, dataset_dir: Path, embed_dir: Path, analyze_embeddings_dir: Path
):
    dataset_names, dataset_params_names = get_list_of_presets(datasets_cfg)
    chunking_names, chunking_params_names = get_list_of_presets(chunking_cfg)
    embed_names, embed_params_names = get_list_of_presets(embed_cfg)


    for (
        (analyze_name, analyze_preset),
        split,
    ) in product(
        iter_cfg_with_presets(analyze_cfg),
        splits,
    ):
        df_embedding = load_multiple_embeddings_dataframes(embed_dir=embed_dir, 
                                                            dataset_names=dataset_names, dataset_params_names=dataset_params_names, 
                                                            chunking_names=chunking_names, chunking_params_names=chunking_params_names, 
                                                            embed_names=embed_names, embed_params_names=embed_params_names, 
                                                            splits=[split])
        df_bookmeta = load_multiple_bookmeta_dataframes(data_dir=dataset_dir, 
                                                        dataset_names=dataset_names, dataset_params_names=dataset_params_names,
                                                        splits=[split])
        analyze_preset_name = analyze_preset["name"]

        analyze_preset_params = analyze_preset["params"]


        result_dir = (
            analyze_embeddings_dir
            / analyze_name
            / analyze_preset_name
            / "-".join(dataset_params_names)
            / "-".join(chunking_names)
            / "-".join(chunking_params_names)
            / "-".join(embed_names)
            / "-".join(embed_params_names)
            / split
        )

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, analyze dir już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"➡️ Szukam query: {result_dir}")
        start = time.perf_counter()  # reset timer
        results = get_analyze_embeddings(
            analyze_name=analyze_name,
            analyze_preset_params=analyze_preset_params,
            result_dir=result_dir,
            df_embedding=df_embedding,
            df_bookmeta=df_bookmeta
        )
        total_time = time.perf_counter() - start
        print(f"✅ Znalazłem query: {result_dir} w czasie {total_time:.2f} sekund")
        
        json_path = result_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, allow_unicode=True)
        with open(result_dir / 'meta.json', "w", encoding="utf-8") as f:
            yaml.safe_dump({"total_time": total_time}, f, allow_unicode=True)


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv("DATASETS_DIR"))
    EMBED_DIR = Path(os.getenv("EMBED_DIR"))
    ANALYZE_EMBEDDINGS_DIR = Path(os.getenv("ANALYZE_EMBEDDINGS_DIR"))
    ANALYZE_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    chunking_cfg = params.get("chunking").get("methods")
    if not chunking_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")

    embed_cfg = params.get("vector_embed").get("methods")
    if not embed_cfg:
        raise ValueError("Nie znaleziono vector_embed_methods w params.yaml")

    analyze_cfg = params.get("analyze_embeddings").get("methods")
    if not analyze_cfg:
        raise ValueError("Nie znaleziono analyze_embeddings_methods w params.yaml")

    splits = params.get("chunking").get("splits")
    if not splits:
        raise ValueError("Nie znaleziono splits w params.yaml")

    analyze_embeddings_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        splits=splits,
        embed_cfg=embed_cfg,
        analyze_cfg=analyze_cfg,
        dataset_dir=DATASET_DIR,
        embed_dir=EMBED_DIR,
        analyze_embeddings_dir=ANALYZE_EMBEDDINGS_DIR,
    )


if __name__ == "__main__":
    main()
