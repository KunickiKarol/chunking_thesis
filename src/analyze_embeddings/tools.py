import numpy as np
from sklearn.preprocessing import normalize

from src.analyze_embeddings.metrics.all_embed_metrics import get_metric


def get_metrics(metrics, normalize_map, X, labels):
    """
    Oblicza wybrane metryki klastrowania dla danych X i etykiet klastrów.

    Parameters:
    - metrics: list of str, nazwy metryk do obliczenia
    - normalize_map: dict, mapa normalizacji
    - X: array-like (n_samples, n_features), dane wejściowe
    - labels: array-like (n_samples,), etykiety klastrów

    Returns:
    - dict: słownik z wynikami metryk
    """

    X = np.asarray(X, dtype=np.float32)
    if any(normalize_map):
        X_normalized = normalize(X, norm="l2", axis=1)

    results = {}
    for metric, should_normalize in zip(metrics, normalize_map):
        if should_normalize:
            metric_result = get_metric(metric, X_normalized, labels)
        else:
            metric_result = get_metric(metric, X, labels)
        results.update(metric_result)
    return results
