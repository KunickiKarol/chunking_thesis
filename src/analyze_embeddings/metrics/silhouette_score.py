from sklearn.metrics import silhouette_score as sk_silhouette_score

from src.analyze_embeddings.metrics.register import register_embed_metric


@register_embed_metric("silhouette_score")
def silhouette_score(X, labels):
    """
    Oblicza silhouette score dla danych X z metryką cosine.

    Parameters:
    - X: array-like (n_samples, n_features)
    - labels: array-like (n_samples,)

    Returns:
    - float: silhouette score
    """
    return {"silhouette_score": sk_silhouette_score(X, labels, metric="cosine")}
