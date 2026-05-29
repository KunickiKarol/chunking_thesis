import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.analyze_embeddings.metrics.register import register_embed_metric


@register_embed_metric("knn_temporal_consistency")
def knn_temporal_consistency(
    X,
    labels,
    k=10,
    metric="cosine",
    normalize=False,
    return_per_sample=False,
    algorithm="auto",
    n_jobs=-1,
):
    """
    Fast vectorized temporal consistency score.

    Lower score = embeddings preserve temporal locality better.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    if len(X) != len(labels):
        raise ValueError("X and labels must have same length")

    n_samples = len(X)

    if k >= n_samples:
        raise ValueError("k must be smaller than number of samples")

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs,
    )

    nn.fit(X)

    _, indices = nn.kneighbors(X)

    # remove self-neighbor
    neighbor_indices = indices[:, 1:]

    # shape: (n_samples, k)
    neighbor_years = labels[neighbor_indices]

    # broadcasting
    temporal_gaps = np.abs(labels[:, None] - neighbor_years)

    # mean gap per sample
    scores = temporal_gaps.mean(axis=1)

    if normalize:
        year_std = labels.std()

        if year_std > 0:
            scores = scores / year_std

    global_score = scores.mean()

    if return_per_sample:
        return global_score, scores

    return {"knn_temporal_consistency": global_score}
