from sklearn.metrics import calinski_harabasz_score

from src.analyze_embeddings.metrics.register import register_embed_metric

@register_embed_metric("calinski_harabasz")
def calinski_harabasz(X, labels):
    """
    Calinski-Harabasz Index (ratio wariancji między a wewnątrz klastrów).
    Wyższe wartości oznaczają lepsze klastry.
    """
    return { "calinski_harabasz": calinski_harabasz_score(X, labels) }