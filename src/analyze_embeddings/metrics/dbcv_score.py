import hdbscan
import numpy as np

from src.analyze_embeddings.metrics.register import register_embed_metric


@register_embed_metric("dbcv_score")
def dbcv_score(X, labels, metric="cosine"):
    """
    DBCV – miara walidacji klastrów gęstościowych (zakres [-1,1]).
    Większa wartość = lepsza jakość klastrów【10†L1212-L1220】.
    """
    X_float64 = np.asarray(X, dtype=np.float64)
    return {"dbcv_score": hdbscan.validity.validity_index(X_float64, labels, metric=metric)}
