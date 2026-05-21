from sklearn.metrics import davies_bouldin_score

from src.analyze_embeddings.metrics.register import register_embed_metric

@register_embed_metric("davies_bouldin")
def davies_bouldin(X, labels):
    """
    Davies-Bouldin Index (DBI). Niższe wartości oznaczają lepsze klastry.
    """
    return { "davies_bouldin": davies_bouldin_score(X, labels) }