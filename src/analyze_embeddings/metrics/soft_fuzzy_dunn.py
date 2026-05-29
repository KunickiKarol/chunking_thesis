import numpy as np
from scipy.spatial.distance import cdist

from src.analyze_embeddings.metrics.register import register_embed_metric
from src.tools.metrics import build_membership_matrix


@register_embed_metric("soft_fuzzy_dunn")
def soft_fuzzy_dunn(
    X,
    labels,
    m=2.0,
    metric="cosine",
    normalize_by_label_count=True,
    eps=1e-12,
):
    X = np.asarray(X, dtype=np.float32)

    U, _ = build_membership_matrix(labels, normalize_by_label_count)

    um = U**m

    centroids = (um.T @ X) / (um.sum(axis=0)[:, None] + eps)

    centroid_dist = cdist(centroids, centroids, metric=metric)
    np.fill_diagonal(centroid_dist, np.inf)

    intercluster = np.min(centroid_dist)

    dist = cdist(X, centroids, metric=metric)

    diameters = []
    for k in range(U.shape[1]):
        denom = np.sum(um[:, k]) + eps
        diam = np.sum(um[:, k] * dist[:, k]) / denom
        diameters.append(diam)

    return {"soft_fuzzy_dunn": float(intercluster / (np.max(diameters) + eps))}
