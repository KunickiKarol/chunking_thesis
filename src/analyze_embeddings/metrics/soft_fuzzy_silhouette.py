import numpy as np
from scipy.spatial.distance import cdist

from src.analyze_embeddings.metrics.register import register_embed_metric
from src.tools.metrics import build_membership_matrix


@register_embed_metric("soft_fuzzy_silhouette")
def soft_fuzzy_silhouette(
    X,
    labels,
    m=2.0,
    alpha=1.0,
    metric="cosine",
    normalize_by_label_count=True,
    eps=1e-12,
):
    X = np.asarray(X, dtype=np.float32)

    U, _ = build_membership_matrix(labels, normalize_by_label_count)

    um = U**m

    centroids = (um.T @ X) / (um.sum(axis=0)[:, None] + eps)

    dist = cdist(X, centroids, metric=metric)

    assigned = np.argmax(U, axis=1)

    a = dist[np.arange(len(X)), assigned]

    dist2 = dist.copy()
    dist2[np.arange(len(X)), assigned] = np.inf
    b = np.min(dist2, axis=1)

    s = (b - a) / (np.maximum(a, b) + eps)

    top2 = np.partition(U, -2, axis=1)[:, -2:]
    weights = (top2[:, 1] - top2[:, 0]) ** alpha

    return {"soft_fuzzy_silhouette": float(np.sum(weights * s) / (np.sum(weights) + eps))}
