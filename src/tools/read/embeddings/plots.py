import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_label_counts(label_counts, figsize=(12, 6)):
    """
    Tworzy wykres liczności klas.

    Parameters
    ----------
    label_counts : dict
        {label: count}
    figsize : tuple
        Rozmiar wykresu.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura gotowa do zapisania.
    ax : matplotlib.axes.Axes
        Oś wykresu.
    """

    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(labels, counts)

    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Label counts")

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, ax


def save_metrics_figure_png(df: pd.DataFrame, path: str = "metrics.png"):
    df = df.copy()

    # tylko kolumny numeryczne (metryki)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # global max/min per kolumna
    col_max = df[numeric_cols].max()
    col_min = df[numeric_cols].min()

    def color_cells(val, col):
        if pd.isna(val):
            return ""

        if val == col_max[col]:
            return "background-color: limegreen; font-weight: bold"
        if val == col_min[col]:
            return "background-color: tomato; font-weight: bold"
        return ""

    def style_df(x):
        styled = pd.DataFrame("", index=x.index, columns=x.columns)

        for col in numeric_cols:
            styled[col] = x[col].apply(lambda v: color_cells(v, col))

        # wyróżnij wiersze MAX / MIN jeśli istnieją
        if "__MAX__" in x.index:
            styled.loc["__MAX__"] = "background-color: #d9fdd3; font-weight: bold"
        if "__MIN__" in x.index:
            styled.loc["__MIN__"] = "background-color: #ffd6d6; font-weight: bold"

        return styled

    styled = df.style.apply(lambda _: style_df(df), axis=None)

    # zapis do PNG
    styled.to_html("temp.html")  # fallback (czasem debug)
    import dataframe_image as dfi

    dfi.export(styled, path)

    logger.info(f"Saved: {path}")


def plot_value_distribution(label_counts, figsize=(10, 6)):
    """
    Pokazuje rozkład wartości z dict {label: value}.
    """

    values = list(label_counts.values())

    fig, ax = plt.subplots(figsize=figsize)

    ax.violinplot(values, showmeans=True, showmedians=True)

    ax.set_ylabel("Values")
    ax.set_title("Distribution of values")

    plt.tight_layout()

    return fig, ax
