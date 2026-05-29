import numpy as np
from sklearn.preprocessing import normalize

from src.analyze_embeddings.metrics.register import register_embed_metric

_MAX_RAM_GB = 10.0  # cap the similarity matrix at this many GB of RAM


def _max_pairs_for_ram(n_features: int, ram_gb: float = _MAX_RAM_GB) -> int:
    """
    Return the maximum number of samples n such that the float32
    similarity matrix  n x n  fits within `ram_gb` gigabytes.

        n^2 * 4 bytes <= ram_gb * 1024^3
        n <= sqrt(ram_gb * 1024^3 / 4)
    """
    max_cells = ram_gb * (1024**3) / 4  # float32 = 4 bytes
    return int(np.floor(np.sqrt(max_cells)))


@register_embed_metric("inter_intra")
def inter_intra(X, labels, balanced: bool = False):
    """
    Compute intra-cluster and inter-cluster cosine similarity.

    Parameters
    ----------
    X : np.ndarray
        Embedding matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    balanced : bool, optional (default=False)
        If False  – global statistics; large clusters dominate inter_mean.
        If True   – per-cluster statistics are averaged with equal weight,
                    so every cluster contributes equally regardless of size.

    Notes
    -----
    Memory: the full cosine-similarity matrix requires O(n²) RAM.
    When n exceeds the threshold derived from _MAX_RAM_GB (≈25 GB for
    float32), the function **randomly sub-samples** rows/columns down to
    that threshold before computing the matrix. The random sample is
    stratified by label so that every cluster keeps a proportional
    representation.

    Returns
    -------
    dict
        {
            "intra_mean"        : float,
            "inter_mean"        : float,
            "separation"        : float,   # intra_mean - inter_mean
            "intra_inter_ratio" : float,   # intra_mean / inter_mean
            "n_used"            : int,     # samples actually used
            "sampled"           : bool,    # True if sub-sampling occurred
        }
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    n = len(X)
    n_max = _max_pairs_for_ram(X.shape[1])
    sampled = False

    # --- stratified sub-sampling if needed -----------------------------------
    if n > n_max:
        sampled = True
        rng = np.random.default_rng(seed=42)
        unique_labels, counts = np.unique(labels, return_counts=True)
        # proportional allocation, at least 1 sample per cluster
        proportions = counts / counts.sum()
        alloc = np.maximum(1, np.round(proportions * n_max).astype(int))
        # trim if rounding pushes total above n_max
        while alloc.sum() > n_max:
            alloc[np.argmax(alloc)] -= 1

        chosen = []
        for lbl, k in zip(unique_labels, alloc):
            idx = np.where(labels == lbl)[0]
            chosen.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        chosen = np.concatenate(chosen)
        X = X[chosen]
        labels = labels[chosen]
        n = len(X)

    # --- L2 normalise → cosine similarity == dot product --------------------
    X = normalize(X, norm="l2", copy=True)

    sim = X @ X.T  # (n, n) float32

    if balanced:
        # per-cluster averages, then macro-average across clusters
        unique_labels = np.unique(labels)
        intra_vals, inter_vals = [], []

        for lbl in unique_labels:
            mask_lbl = labels == lbl

            # intra: pairs within this cluster (upper triangle only)
            idx = np.where(mask_lbl)[0]
            if len(idx) >= 2:
                sub = sim[np.ix_(idx, idx)]
                # upper triangle without diagonal
                ut = sub[np.triu_indices(len(idx), k=1)]
                if ut.size > 0:
                    intra_vals.append(ut.mean())

            # inter: this cluster vs all others
            idx_other = np.where(~mask_lbl)[0]
            if len(idx) >= 1 and len(idx_other) >= 1:
                cross = sim[np.ix_(idx, idx_other)]
                inter_vals.append(cross.mean())

        intra_mean = float(np.mean(intra_vals)) if intra_vals else float("nan")
        inter_mean = float(np.mean(inter_vals)) if inter_vals else float("nan")

    else:
        # global statistics (original behaviour)
        same = labels[:, None] == labels[None, :]
        diff = ~same
        np.fill_diagonal(same, False)

        intra_mean = float(sim[same].mean()) if same.any() else float("nan")
        inter_mean = float(sim[diff].mean()) if diff.any() else float("nan")

    separation = intra_mean - inter_mean if not (np.isnan(intra_mean) or np.isnan(inter_mean)) else float("nan")
    ratio = intra_mean / (inter_mean + 1e-8) if not (np.isnan(intra_mean) or np.isnan(inter_mean)) else float("nan")

    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "separation": separation,
        "intra_inter_ratio": ratio,
        "n_used": n,
        "sampled": sampled,
    }
