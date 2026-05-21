
import numpy as np

def build_membership_matrix(labels, normalize_by_label_count=True):
    """
    labels: list of lists
    """

    all_labels = sorted(set(l for row in labels for l in row))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    n = len(labels)
    c = len(all_labels)

    U = np.zeros((n, c), dtype=np.float32)

    for i, row in enumerate(labels):
        if len(row) == 0:
            continue

        if normalize_by_label_count:
            w = 1.0 / len(row)
        else:
            w = 1.0

        for l in row:
            U[i, label_to_idx[l]] = w

    return U, all_labels