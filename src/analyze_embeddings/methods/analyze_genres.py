import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src.analyze_embeddings.dim_reduction_plots import get_dim_reduction_plots_multilabel
from src.analyze_embeddings.methods.register import register_analyze_embeddings
from src.analyze_embeddings.tools import get_metrics
from src.tools.read.embeddings.plots import plot_label_counts

logger = logging.getLogger(__name__)


def _parse_genre_tags(genre_str: str) -> list[str]:
    """Split 'Short story;Literary fiction' → ['Short story', 'Literary fiction']."""
    if pd.isna(genre_str) or genre_str == "":
        return []
    return [g.strip() for g in genre_str.split(";") if g.strip()]


def _build_multilabel_matrix(
    genre_series: pd.Series,
) -> tuple[np.ndarray, dict, dict]:
    """
    Build a binary (N, C) indicator matrix from a Series of ';'-separated tags.

    Returns
    -------
    labels_binary : (N, C)
    label_to_id   : {str -> int}
    id_to_label   : {int -> str}
    """
    all_genres_per_row = genre_series.apply(_parse_genre_tags)

    # sorted vocabulary
    vocab = sorted({g for genres in all_genres_per_row for g in genres})
    label_to_id = {g: i for i, g in enumerate(vocab)}
    id_to_label = {i: g for g, i in label_to_id.items()}

    N, C = len(genre_series), len(vocab)
    labels_binary = np.zeros((N, C), dtype=np.int8)
    for row_idx, genres in enumerate(all_genres_per_row):
        for g in genres:
            labels_binary[row_idx, label_to_id[g]] = 1

    return labels_binary, label_to_id, id_to_label


# ── main analysis function ─────────────────────────────────────────────────


@register_analyze_embeddings("analyze_genres")
def analyze_genres(analyze_preset_params, df_embedding, df_bookmeta):
    df_merged = df_embedding.join(
        df_bookmeta,
        how="left",
        validate="many_to_one",
    )
    df_merged = df_merged[df_merged.index.get_level_values("dataset_name") == "literaryQA"]

    group_cols = [
        "chunking_name",
        "chunking_params_name",
        "embed_name",
        "embed_params_name",
    ]

    results = {}
    print(df_merged.shape, df_embedding.shape, df_bookmeta.shape)
    grouped = df_merged.groupby(level=group_cols)
    if not grouped.ngroups  :
        print("puste")  
    for group_keys, group_df in grouped:
        print("DUPA")
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

        # ── multilabel: build binary matrix from genre_tags ────────────────
        labels_binary, label_to_id, id_to_label = _build_multilabel_matrix(group_df["genre_tags"])
        # labels_binary : (N, C)  — used for plots and label_counts


        # ── metrics (use primary label — single-label proxy) ───────────────
        metrics_names = analyze_preset_params["metrics"]
        normalize_map = analyze_preset_params.get("normalize_map")
        metrics = get_metrics(metrics_names, normalize_map, X, labels_binary)

        # ── label counts — per individual genre across all rows ────────────
        unique_ids, counts = np.unique(np.where(labels_binary)[1], return_counts=True)
        label_counts = {id_to_label[i]: int(count) for i, count in zip(unique_ids, counts)}
        fig_num_per_class = plot_label_counts(label_counts)[0]

        # ── dimensionality-reduction plots (one-vs-rest per genre) ─────────
        dim_red_methods = analyze_preset_params.get("dim_red") or []
        dim_red_plots: list[plt.Figure] = []
        dim_red_titles: list[str] = []

        if dim_red_methods:
            dim_red_plots, dim_red_titles = get_dim_reduction_plots_multilabel(
                method_names=dim_red_methods,
                X=X,
                labels_binary=labels_binary,
                id_to_label=id_to_label,
            )

        results[key] = {
            "metrics": metrics,
            "label_counts": label_counts,
            "fig_num_per_class": fig_num_per_class,
            "plots": dim_red_plots,  # list[Figure]  len = n_methods * n_classes
            "plot_titles": dim_red_titles,
        }

    # ── summary table ──────────────────────────────────────────────────────
    best_metrics: dict = {}
    metrics_names = list(metrics.keys()) if metrics else []

    rows: dict = {}
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

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.loc["__MAX__"] = global_max
    df.loc["__MIN__"] = global_min
    metrics_figure = df

    for metric_name in metrics_names:
        best_max_key, best_max_value = None, -np.inf
        best_min_key, best_min_value = None, np.inf

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
