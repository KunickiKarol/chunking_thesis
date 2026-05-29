import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import mantel
from src.analyze_embeddings.metrics.register import register_embed_metric

@register_embed_metric("mantel_test")
def mantel_test(X, labels, embedding_metric="cosine", permutations=9999, max_samples=10000):
    """
    Sprawdza czy książki bliskie w czasie mają podobne embeddingi.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    labels : ndarray (n_samples,)
    embedding_metric : str
    permutations : int
    max_samples : int
        Cap on samples before subsampling. A 5000×5000 matrix ~ 200 MB.
    """
    n = len(X)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        X, labels = X[idx], np.asarray(labels)[idx]

    D_emb   = squareform(pdist(X, metric=embedding_metric))
    years   = np.asarray(labels).reshape(-1, 1)
    D_years = squareform(pdist(years, metric="euclidean"))

    corr, p_value, _ = mantel(
        D_emb, D_years, method="spearman", permutations=permutations
    )
    return {"mantel_corr": corr, "mantel_p_value": p_value}