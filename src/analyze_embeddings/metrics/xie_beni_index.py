import numpy as np
from scipy.spatial.distance import cdist

from src.analyze_embeddings.metrics.register import register_embed_metric
from src.tools.metrics import build_membership_matrix

@register_embed_metric("xie_beni_index")
def xie_beni_index(
    X,
    labels,
    m=2.0,
    metric="cosine",
    normalize_by_label_count=True,
    eps=1e-12,
):
    X = np.asarray(X, dtype=np.float32)

    U, _ = build_membership_matrix(labels, normalize_by_label_count)

    um = U ** m

    centroids = (um.T @ X) / (um.sum(axis=0)[:, None] + eps)

    dist = cdist(X, centroids, metric=metric)

    numerator = np.sum(um * (dist ** 2))

    centroid_dist = cdist(centroids, centroids, metric=metric)
    np.fill_diagonal(centroid_dist, np.inf)

    min_sep = np.min(centroid_dist) ** 2

    return { "xie_beni_index": float(numerator / (len(X) * min_sep + eps)) }