import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import mantel
from src.analyze_embeddings.metrics.register import register_embed_metric

@register_embed_metric("mantel_test")       
def mantel_test(
    X,
    labels,
    embedding_metric="cosine",
    permutations=9999
):
    """
    Sprawdza czy książki bliskie w czasie
    mają podobne embeddingi.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Embeddingi książek

    labels : ndarray (n_samples,)
        Lata publikacji

    embedding_metric : str
        np. 'cosine', 'euclidean'

    permutations : int
        liczba permutacji

    Returns
    -------
    corr : float
        Korelacja Mantela

    p_value : float
        Istotność statystyczna
    """

    # macierz odległości embeddingów
    D_emb = squareform(
        pdist(X, metric=embedding_metric)
    )

    # macierz odległości lat
    years = np.asarray(labels).reshape(-1, 1)

    D_years = squareform(
        pdist(years, metric="cosine")
    )

    corr, p_value, n = mantel(
        D_emb,
        D_years,
        method="spearman",
        permutations=permutations
    )

    return {'mantel_corr': corr, 'mantel_p_value': p_value}