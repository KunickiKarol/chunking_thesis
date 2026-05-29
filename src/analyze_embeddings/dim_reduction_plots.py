import logging
import os
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from src.analyze_embeddings.dim_reduction import (
    reduce_isomap,
    reduce_pacmap,
    reduce_pca,
    reduce_phate,
    reduce_tsne,
    reduce_umap,
)

logger = logging.getLogger(__name__)


_REDUCERS = {
    "pca": reduce_pca,
    "tsne": reduce_tsne,
    "umap": reduce_umap,
    "isomap": reduce_isomap,
    "phate": reduce_phate,
    "pacmap": reduce_pacmap,
}


def _make_color_map(unique_labels):
    """
    Return a dict {label: color} using visually distinct colors.
    """

    # bardzo różnorodne kolory dla małej liczby klas
    base_colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    n = len(unique_labels)

    if n <= len(base_colors):
        colors = base_colors[:n]
    else:
        cmap = cm.get_cmap("tab20", n)
        colors = [cmap(i) for i in range(n)]

    return {label: color for label, color in zip(unique_labels, colors)}


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    id_to_label: dict,
) -> plt.Figure:
    """
    Generic 2-D scatter plot with a legend showing real label names.

    Parameters
    ----------
    coords    : (N, 2) array of reduced coordinates
    labels    : (N,) array of integer label ids
    title     : figure title
    id_to_label : mapping from integer id back to human-readable label name
    """
    unique_ids = np.unique(labels)
    unique_names = [id_to_label[i] for i in unique_ids]
    color_map = _make_color_map(unique_names)

    fig, ax = plt.subplots(figsize=(10, 7))
    for uid, name in zip(unique_ids, unique_names):
        mask = labels == uid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=name,
            color=color_map[name],
            alpha=0.7,
            s=18,
            linewidths=0,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(
        title="Label",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig


def get_dim_reduction_plots_one_rest(
    method_names: list[str],
    X: np.ndarray,
    label_ids: np.ndarray,
    id_to_label: dict,
    random_state: Optional[int] = None,
    max_figs: int = 5,
) -> tuple[list[plt.Figure], list[str]]:
    """
    Return figures and titles for one-vs-rest dimensionality reduction plots.

    For every reduction method and every label:
        - creates a binary one-vs-rest label vector
        - generates a figure
        - appends a title:
            f"{METHOD_NAME} — {CLASS_NAME}"

    Parameters
    ----------
    method_names : e.g. ["pca", "tsne", "umap"]
    X            : (N, D) embedding matrix
    label_ids    : (N,) integer label ids
    id_to_label  : {int -> str} mapping used for class names
    random_state : forwarded to every reducer that accepts it
    max_figs     : maximum number of returned figures

    Returns
    -------
    figs   : list[plt.Figure]
    titles : list[str]
    """
    if random_state is None:
        random_state = int(os.getenv("RANDOM_SEED", 42))

    figs = []
    titles = []

    unique_labels = np.unique(label_ids)

    for name in method_names:
        name_lower = name.lower()

        if name_lower not in _REDUCERS:
            raise ValueError(f"Unknown dim-reduction method '{name}'. " f"Available: {list(_REDUCERS)}")

        reducer = _REDUCERS[name_lower]

        for label_id in unique_labels:
            if len(figs) >= max_figs:
                return figs, titles

            class_name = id_to_label.get(label_id, str(label_id))

            # one-vs-rest labels
            binary_labels = (label_ids == label_id).astype(int)

            # binary mapping for legend/display
            binary_id_to_label = {
                0: "rest",
                1: class_name,
            }

            coords = reducer(X, random_state)
            fig = _scatter_plot(coords, binary_labels, f"{name.upper()} — {class_name}", binary_id_to_label)

            figs.append(fig)
            titles.append(f"{name.upper()} — {class_name}")

    return figs, titles


def get_dim_reduction_plots_multilabel(
    method_names: list[str],
    X: np.ndarray,
    labels_binary: np.ndarray,  # (N, C) binary matrix
    id_to_label: dict,  # {int -> str}  C classes
    random_state: Optional[int] = None,
) -> tuple[list[plt.Figure], list[str]]:
    """
    One-vs-rest dimensionality reduction plots for multilabel data.

    For each (method, class) pair, a separate figure is produced where
    points belonging to the class are highlighted against the rest.

    Parameters
    ----------
    method_names   : e.g. ["pca", "tsne", "umap"]
    X              : (N, D) embedding matrix
    labels_binary  : (N, C) binary indicator matrix
    id_to_label    : {int -> str} mapping for C classes
    random_state   : forwarded to every reducer that accepts it

    Returns
    -------
    figs   : list of matplotlib Figure objects
    titles : list of matching title strings  (method — ClassName)
    """
    if random_state is None:
        random_state = int(os.getenv("RANDOM_SEED", 42))

    n_classes = labels_binary.shape[1]
    figs: list[plt.Figure] = []
    titles: list[str] = []

    for name in method_names:
        name_lower = name.lower()
        if name_lower not in _REDUCERS:
            raise ValueError(f"Unknown dim-reduction method '{name}'. " f"Available: {list(_REDUCERS)}")
        reducer = _REDUCERS[name_lower]

        # Compute 2-D projection ONCE per method (unsupervised — no labels).
        # We pass a dummy all-zeros label so the reducer just projects,
        # then we re-colour per class below.
        dummy_labels = np.zeros(len(X), dtype=int)
        dummy_id_to_label = {0: "all"}
        coords = reducer(X, random_state)  # (N, 2)
        base_fig = _scatter_plot(coords, dummy_labels, f"{name} — all", dummy_id_to_label)


        # Extract the 2-D coordinates from the base figure's axes.
        ax_base = base_fig.axes[0]
        # All scatter points are in the first PathCollection
        xy = ax_base.collections[0].get_offsets().data  # (N, 2)
        plt.close(base_fig)

        # One figure per class — one-vs-rest colouring
        limit_counter = 0
        for class_idx in range(n_classes):
            class_name = id_to_label[class_idx]
            is_member = labels_binary[:, class_idx].astype(bool)

            fig, ax = plt.subplots(figsize=(8, 6))

            # "rest" in grey
            ax.scatter(
                xy[~is_member, 0],
                xy[~is_member, 1],
                c="#249e94",
                s=10,
                alpha=0.4,
                label="rest",
            )
            # class members in colour
            ax.scatter(
                xy[is_member, 0],
                xy[is_member, 1],
                c="#e05c2a",
                s=18,
                alpha=0.85,
                label=class_name,
            )

            title = f"{name.upper()} — {class_name}"
            ax.set_title(title, fontsize=11)
            ax.legend(loc="best", fontsize=8, markerscale=1.5)
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            fig.tight_layout()

            figs.append(fig)
            titles.append(title)
            limit_counter += 1
            if limit_counter >= 20:
                logger.warning(
                    f"Warning: more than 20 classes in multilabel data. " f"Only the first 20 will be plotted."
                )
                break

    return figs, titles





def _plot_continuous_years(
    coords: np.ndarray,
    years_float: np.ndarray,
    title: str,
) -> plt.Figure:
    """
    Scatter plot with a smooth colormap over year values.

    Parameters
    ----------
    coords      : (N, 2) reduced coordinates
    years_float : (N,) float array of publication years
    title       : figure title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=years_float,
        cmap="plasma",
        alpha=0.75,
        s=18,
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Year", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig


def _plot_binned_years(
    coords: np.ndarray,
    years_float: np.ndarray,
    title: str,
    bin_size: int = 30,
) -> plt.Figure:
    """
    Scatter plot where points are coloured by year-bin of width `bin_size`,
    bins defined from the newest year downward.

    Parameters
    ----------
    coords      : (N, 2) reduced coordinates
    years_float : (N,) float array of publication years
    title       : figure title
    bin_size    : width of each year bin (default 30)
    """
    year_max = int(np.max(years_float))
    year_min = int(np.min(years_float))

    # Build bin edges from newest year downward so that
    # the most recent bin is always "clean" (e.g. 1990–2020, 1960–1990, ...)
    edges = list(range(year_max, year_min - bin_size, -bin_size))
    # ensure we cover the oldest point
    if edges[-1] > year_min:
        edges.append(edges[-1] - bin_size)
    edges = sorted(edges)  # ascending for np.digitize

    # assign each point to a bin index (0 = oldest)
    bin_ids = np.digitize(years_float, bins=edges, right=True)
    # np.digitize may return 0 for values below edges[0] — clamp
    bin_ids = np.clip(bin_ids, 0, len(edges) - 1)

    unique_bins = sorted(np.unique(bin_ids))
    n_bins = len(unique_bins)

    cmap = cm.get_cmap("tab20", max(n_bins, 1))
    color_map = {b: cmap(i) for i, b in enumerate(unique_bins)}

    # Build readable labels: "YYYY – YYYY"
    def _bin_label(b: int) -> str:
        lo = edges[b - 1] if b > 0 else year_min
        hi = edges[b] if b < len(edges) else year_max
        return f"{int(lo)+1}–{int(hi)}"

    fig, ax = plt.subplots(figsize=(11, 7))

    for b in unique_bins:
        mask = bin_ids == b
        label = _bin_label(b)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=color_map[b],
            label=label,
            alpha=0.75,
            s=18,
            linewidths=0,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(
        title=f"{bin_size}-year bins",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig


def get_dim_reduction_plots_temporal(
    method_names: list[str],
    X: np.ndarray,
    label_ids: np.ndarray,
    id_to_label: dict,
    random_state: Optional[int] = None,
    bin_size: int = 30,
) -> tuple[list[plt.Figure], list[str]]:
    """
    Dimensionality reduction plots with temporal (year) colouring.

    For each reduction method two figures are produced:
        1. Continuous gradient — smooth colormap over raw year values.
        2. Binned — points grouped into `bin_size`-year windows
           starting from the newest year and going backward.

    The reduction is computed **once** per method; both figures share
    the same 2-D projection.

    Parameters
    ----------
    method_names : e.g. ["pca", "tsne", "umap"]
    X            : (N, D) embedding matrix
    label_ids    : (N,) integer label ids whose string values are
                   publication years (resolved via id_to_label)
    id_to_label  : {int -> str} mapping; values must be parseable as float years
    random_state : forwarded to every reducer; falls back to RANDOM_SEED env var
    bin_size     : width of each year bin for the binned plot (default 30)

    Returns
    -------
    figs   : list[plt.Figure]  — length = 2 * len(method_names)
             order: [method0_continuous, method0_binned,
                     method1_continuous, method1_binned, ...]
    titles : list[str]         — matching titles with suffixes
                                 "(continuous)" and f"({bin_size}-year bins)"
    """
    if random_state is None:
        random_state = int(os.getenv("RANDOM_SEED", 42))

    # Resolve integer label ids → float years
    try:
        years_float = np.array(
            [float(id_to_label[i]) for i in label_ids],
            dtype=float,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "id_to_label values must be parseable as float year values. " f"Original error: {exc}"
        ) from exc

    figs: list[plt.Figure] = []
    titles: list[str] = []

    for name in method_names:
        name_lower = name.lower()
        if name_lower not in _REDUCERS:
            raise ValueError(f"Unknown dim-reduction method '{name}'. " f"Available: {list(_REDUCERS)}")

        # --- single reduction call shared by both plots ---
        try:
            coords = _REDUCERS[name_lower](X, random_state)
        except Exception:
            logger.exception("Reduction '%s' failed — skipping.", name)
            continue

        method_upper = name.upper()

        # 1. Continuous
        title_cont = f"{method_upper} (continuous)"
        figs.append(_plot_continuous_years(coords, years_float, title_cont))
        titles.append(title_cont)

        # 2. Binned
        title_bin = f"{method_upper} ({bin_size}-year bins)"
        figs.append(_plot_binned_years(coords, years_float, title_bin, bin_size=bin_size))
        titles.append(title_bin)

    return figs, titles


def get_dim_reduction_plots(
    method_names: list[str],
    X: np.ndarray,
    labels: np.ndarray,
    id_to_label: dict,
    random_state: Optional[int] = None,
    max_figs: int = 20,
) -> tuple[list[plt.Figure], list[str]]:
    """
    Return figures and titles for one-vs-rest dimensionality reduction plots.

    For every reduction method and every label:
        - creates a binary one-vs-rest label vector
        - generates a figure
        - appends a title:
            f"{METHOD_NAME} — {CLASS_NAME}"

    Parameters
    ----------
    method_names : e.g. ["pca", "tsne", "umap"]
    X            : (N, D) embedding matrix
    labels       : (N,) integer label ids
    id_to_label  : {int -> str} mapping used for class names
    random_state : forwarded to every reducer that accepts it
    max_figs     : maximum number of returned figures

    Returns
    -------
    figs   : list[plt.Figure]
    titles : list[str]
    """
    if random_state is None:
        random_state = int(os.getenv("RANDOM_SEED", 42))
    figs = []
    titles = []

    unique_labels = np.unique(labels)

    for name in method_names:
        logger.info(f"Generating dim-reduction plots for method '{name}'")
        name_lower = name.lower()

        if name_lower not in _REDUCERS:
            raise ValueError(
                f"Unknown dim-reduction method '{name}'. "
                f"Available: {list(_REDUCERS)}"
            )

        reducer = _REDUCERS[name_lower]

        for label_id in unique_labels:
            if len(figs) >= max_figs:
                return figs, titles

            class_name = id_to_label.get(label_id, str(label_id))

            # one-vs-rest labels
            binary_labels = (labels == label_id).astype(int)

            # binary mapping for legend/display
            binary_id_to_label = {
                0: "rest",
                1: class_name,
            }
            coords = reducer(X, random_state)
            fig = _scatter_plot(coords, binary_labels, f"{name} — {class_name}", binary_id_to_label)

            figs.append(fig)
            titles.append(f"{name.upper()} — {class_name}")

    return figs, titles