import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from src.analyze_embeddings.metrics.register import register_embed_metric


@register_embed_metric("ridge")
def ridge(X, labels):
    r2_scores = cross_val_score(Ridge(alpha=1.0), X, labels, cv=5, scoring="r2")
    mae_scores = -cross_val_score(Ridge(alpha=1.0), X, labels, cv=5, scoring="neg_mean_absolute_error")

    r2 = float(np.mean(r2_scores))
    r2_mae = float(np.mean(mae_scores))
    return {"year_r2": r2, "year_mae": r2_mae}
