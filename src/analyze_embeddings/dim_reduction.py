import logging
import os
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def reduce_pca(
    X: np.ndarray,
    random_state: int = 42,
) -> plt.Figure:
    """PCA reduction to 2-D using cosine-equivalent normalisation (L2 + PCA)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    logger.info(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_[:2])} for 2 components")
    return coords


def reduce_tsne(
    X: np.ndarray,
    random_state: int = 42,
) -> plt.Figure:
    """t-SNE with cosine metric."""
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, len(X) // 10))
    coords = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
        n_jobs=-1,
    ).fit_transform(X)
    return coords


def reduce_umap(
    X: np.ndarray,
    random_state: int = 42,
) -> plt.Figure:
    """UMAP with cosine metric."""
    import umap  # pip install umap-learn

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=random_state,
        n_jobs=-1,
    )
    coords = reducer.fit_transform(X)
    return coords


def reduce_isomap(
    X: np.ndarray,
    random_state: int = 42,  # unused, kept for uniform signature
) -> plt.Figure:
    """Isomap – uses precomputed cosine-distance matrix."""
    from sklearn.manifold import Isomap
    from sklearn.metrics.pairwise import cosine_distances

    dist = cosine_distances(X)
    n_neighbors = min(10, len(X) - 1)
    coords = Isomap(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="precomputed",
    ).fit_transform(dist)
    return coords


def reduce_phate(
    X: np.ndarray,
    random_state: int = 42,
) -> plt.Figure:
    """PHATE with cosine knn distance."""
    import phate  # pip install phate

    phate_op = phate.PHATE(
        n_components=2,
        knn_dist="cosine",
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    coords = phate_op.fit_transform(X)
    return coords


def reduce_pacmap(
    X: np.ndarray,
    random_state: int = 42,
) -> plt.Figure:
    """PaCMAP dimensionality reduction."""
    import pacmap  # pip install pacmap

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=None, random_state=random_state, distance="angular")
    coords = reducer.fit_transform(X, init="pca")
    return coords
