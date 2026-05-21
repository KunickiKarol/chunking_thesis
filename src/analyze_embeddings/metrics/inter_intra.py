import numpy as np
from sklearn.preprocessing import normalize

#register_embed_metric is imported in the same way as in other metric files
from src.analyze_embeddings.metrics.register import register_embed_metric

@register_embed_metric("inter_intra")
def inter_intra(X, labels):
    """
    Compute intra-cluster and inter-cluster cosine similarity.

    Parameters
    ----------
    X : np.ndarray
        Embedding matrix of shape (n_samples, n_features)

    labels : np.ndarray
        Cluster labels of shape (n_samples,)

    Returns
    -------
    dict
        {
            "intra_mean": float,
            "inter_mean": float,
            "separation": float,
            "intra_inter_ratio": float
        }
    """

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)


    # L2 normalize -> cosine similarity = dot product
    X = normalize(X, norm="l2", copy=False)

    # cosine similarity matrix
    sim = X @ X.T

    # masks
    same = labels[:, None] == labels[None, :]
    diff = ~same

    # remove diagonal (self similarity)
    np.fill_diagonal(same, False)

    intra_mean = sim[same].mean()
    inter_mean = sim[diff].mean()

    return {
        "intra_mean": float(intra_mean),
        "inter_mean": float(inter_mean),
        "separation": float(intra_mean - inter_mean),
        "intra_inter_ratio": float(intra_mean / (inter_mean + 1e-8)),
    }