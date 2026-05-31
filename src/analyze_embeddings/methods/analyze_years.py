import logging
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.analyze_embeddings.dim_reduction_plots import get_dim_reduction_plots_temporal
from src.analyze_embeddings.methods.register import register_analyze_embeddings
from src.analyze_embeddings.tools import get_metrics
from src.tools.read.embeddings.plots import plot_histogram, plot_label_counts

logger = logging.getLogger(__name__)


@register_analyze_embeddings("analyze_years")
def analyze_years(analyze_preset_params, df_embedding, df_bookmeta):
    df_merged = df_embedding.join(
        df_bookmeta,
        how="left",
        validate="many_to_one",
    )
    for col in ["publication_date", "yearpub"]:
        if col not in df_merged.columns:
            df_merged[col] = np.nan

    df_merged = df_merged[
        df_merged.index.get_level_values("dataset_name").isin(["novelQA", "literaryQA"])
    ]

    df_merged = df_merged[
        ~(
            df_merged["yearpub"].isna() &
            df_merged["publication_date"].isna()
        )
    ]

    df_merged = df_merged[
        df_merged["publication_date"] != "Unknown"
    ]

    group_cols = [
        "chunking_name",
        "chunking_params_name",
        "embed_name",
        "embed_params_name",
    ]

    results = {}
    grouped = df_merged.groupby(level=group_cols)

    for group_keys, group_df in grouped:
        (
            chunking_name,
            chunking_params_name,
            embed_name,
            embed_params_name,
        ) = group_keys

        key = f"{chunking_name}-" f"{chunking_params_name}-" f"{embed_name}-" f"{embed_params_name}"

        X = np.stack(group_df["embedding"].values)

        # ── optional PCA pre-reduction (for metrics only) ──────────────────
        if analyze_preset_params.get("pca_dim", 0) > 0:
            pca = PCA(
                n_components=analyze_preset_params["pca_dim"],
                random_state=int(os.getenv("RANDOM_SEED", 42)),
            )
            X = pca.fit_transform(X)
            logger.info(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_[:analyze_preset_params['pca_dim']])} for {analyze_preset_params['pca_dim']} components")
        def safe_int(x):
            try:
                if pd.isna(x):
                    return None

                x = str(x).strip()

                # case 1: "L.M something 1828"
                if x.startswith("L.M"):
                    x = x.split()[-1]

                # case 2: "1853-1857"
                if "-" in x:
                    x = x.split("-")[0]

                return int(float(x))

            except:
                return None
        labels = [
            safe_int(y) if pd.notna(y) and y != "ann" else safe_int(p)
            for y, p in zip(group_df["yearpub"], group_df["publication_date"])
        ]
        label_to_id = {label: i for i, label in enumerate(sorted(set(labels)))}
        id_to_label = {v: k for k, v in label_to_id.items()}
        labels_id = np.array(
            [label_to_id[label] for label in labels],
            dtype=int,
        )

        # ── metrics ────────────────────────────────────────────────────────
        metrics_names = analyze_preset_params["metrics"]
        normalize_map = analyze_preset_params.get("normalize_map")
        metrics = get_metrics(metrics_names, normalize_map, X, labels_id)

        # ── label counts plot ──────────────────────────────────────────────
        unique_ids, counts = np.unique(labels_id, return_counts=True)
        label_counts = {id_to_label[i]: int(count) for i, count in zip(unique_ids, counts)}
        fig_num_per_class = plot_histogram(label_counts)[0]

        # ── dimensionality-reduction plots ─────────────────────────────────
        dim_red_methods = analyze_preset_params.get("dim_red") or []
        dim_red_plots = []
        plot_titles = []
        if dim_red_methods:
            dim_red_plots, plot_titles = get_dim_reduction_plots_temporal(
                method_names=dim_red_methods,
                X=X,
                label_ids=labels_id,
                id_to_label=id_to_label,
                bin_size=analyze_preset_params.get("year_bin_size", 30),
            )
        results[key] = {
            "metrics": metrics,
            "label_counts": label_counts,
            "fig_num_per_class": fig_num_per_class,
            "plots": dim_red_plots,  # list[matplotlib.figure.Figure]
            "plot_titles": plot_titles,
        }

    best_metrics = {}
    metrics_names = list(metrics.keys()) if metrics else []

    # tabela: rows=result_key, cols=metric_name
    rows = {}

    # do wyliczenia globalnych min/max per metryka (na potrzeby dodatkowych wierszy)
    global_max = {m: -np.inf for m in metrics_names}
    global_min = {m: np.inf for m in metrics_names}

    for result_key, result_data in results.items():
        rows[result_key] = {}

        for metric_name in metrics_names:
            metric_value = result_data["metrics"].get(metric_name)

            rows[result_key][metric_name] = metric_value

            if metric_value is None:
                continue

            global_max[metric_name] = max(global_max[metric_name], metric_value)
            global_min[metric_name] = min(global_min[metric_name], metric_value)

    # normalna tabela
    df = pd.DataFrame.from_dict(rows, orient="index")

    if global_max:
        df.loc["__MAX__"] = global_max
    if global_min: 
        df.loc["__MIN__"] = global_min


    metrics_figure = df

    # BEST METRICS (zostaje jak było)
    for metric_name in metrics_names:
        best_max_key = None
        best_max_value = -np.inf

        best_min_key = None
        best_min_value = np.inf

        for result_key, result_data in results.items():
            metric_value = result_data["metrics"].get(metric_name)

            if metric_value is None:
                continue

            if metric_value > best_max_value:
                best_max_value = metric_value
                best_max_key = result_key

            if metric_value < best_min_value:
                best_min_value = metric_value
                best_min_key = result_key

        best_metrics[metric_name] = {
            "max": (best_max_key, best_max_value),
            "min": (best_min_key, best_min_value),
        }

    return results, best_metrics, metrics_figure
