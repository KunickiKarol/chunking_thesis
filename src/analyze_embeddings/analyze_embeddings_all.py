#!/usr/bin/env python
import json
import logging
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from src.analyze_embeddings.methods.all_analyze_embeddings import get_analyze_embeddings
from src.tools.logging_config import setup_logging
from src.tools.presets import get_list_of_presets, iter_cfg_with_presets
from src.tools.read.datasets.load_metadata import load_multiple_bookmeta_dataframes
from src.tools.read.embeddings.load_embeddings import load_multiple_embeddings_dataframes
from src.tools.read.embeddings.plots import save_metrics_figure_png

setup_logging()
logger = logging.getLogger(__name__)


def analyze_embeddings_all(
    datasets_cfg,
    chunking_cfg,
    splits,
    embed_cfg,
    analyze_cfg,
    dataset_dir: Path,
    embed_dir: Path,
    analyze_embeddings_dir: Path,
):
    dataset_names, dataset_params_names = get_list_of_presets(datasets_cfg)
    chunking_names, chunking_params_names = get_list_of_presets(chunking_cfg)
    embed_names, embed_params_names = get_list_of_presets(embed_cfg)

    for (
        (analyze_name, analyze_preset),
        # (chunking_names, chunking_params_names),
        split,
    ) in product(
        iter_cfg_with_presets(analyze_cfg),
        # iter_cfg_with_presets(chunking_cfg),
        splits,
    ):
        # chunking_names = [chunking_names]
        # chunking_params_names = [chunking_params_names['name']]
        df_embedding = load_multiple_embeddings_dataframes(
            embed_dir=embed_dir,
            dataset_names=dataset_names,
            dataset_params_names=dataset_params_names,
            chunking_names=chunking_names,
            chunking_params_names=chunking_params_names,
            embed_names=embed_names,
            embed_params_names=embed_params_names,
            split_names=[split],
        )
        df_bookmeta = load_multiple_bookmeta_dataframes(
            data_dir=dataset_dir, dataset_names=dataset_names, dataset_params_names=dataset_params_names
        )
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
        if "local" in str(result_dir):
            logger.info(f"⏭️ Pomijam {result_dir}, zawiera 'local'")

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            logger.info(f"⏭️ Pomijam {result_dir}, analyze dir już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"➡️ Szukam query: {result_dir}")
        start = time.perf_counter()  # reset timer
        results, best_metrics, metrics_figure = get_analyze_embeddings(
            analyze_embedding_name=analyze_preset_params["analyze_type"],
            analyze_preset_params=analyze_preset_params,
            df_embedding=df_embedding,
            df_bookmeta=df_bookmeta,
        )
        total_time = time.perf_counter() - start
        logger.info(f"✅ Analyzed embeddings: {result_dir} w czasie {total_time:.2f} sekund")

        # ── JSON results ─────────────────────────────────────────────────────────
        json_path = result_dir / "results.json"

        def to_jsonable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        results_json = {
            k: {
                kk: to_jsonable(vv)
                for kk, vv in v.items()
                if kk not in ("fig_num_per_class", "plots", "plot_titles")  # nie zapisujemy figur do JSON
            }
            for k, v in results.items()
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)

        # ── meta ────────────────────────────────────────────────────────────────
        with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"total_time": total_time, "best_metrics": best_metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ── plots per key ───────────────────────────────────────────────────────
        plots_root = result_dir / "plots"
        plots_root.mkdir(parents=True, exist_ok=True)

        for key, data in results.items():
            key_dir = plots_root / key
            key_dir.mkdir(parents=True, exist_ok=True)

            # label counts plot
            fig = data.get("fig_num_per_class")
            if fig is not None:
                fig.savefig(key_dir / "label_counts.png", dpi=300, bbox_inches="tight")

            # dimensionality reduction plots
            plot_titles = data.get("plot_titles")
            for i, fig in enumerate(data.get("plots", [])):
                if plot_titles and i < len(plot_titles):
                    safe_title = plot_titles[i].replace(" ", "_").replace("—", "-").replace("/", "-")
                    filename = f"{safe_title}.png"
                else:
                    filename = f"dim_reduction_{i}.png"
                fig.savefig(key_dir / filename, dpi=300, bbox_inches="tight")

        save_metrics_figure_png(metrics_figure, path=result_dir / "metrics.png")


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
