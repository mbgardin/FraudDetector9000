import numpy as np

def minmax_normalize(scores, ref_min=None, ref_max=None):
    scores = np.asarray(scores, dtype=float)

    if ref_min is None:
        ref_min = scores.min()
    if ref_max is None:
        ref_max = scores.max()

    if ref_max <= ref_min:
        return np.zeros_like(scores)

    return (scores - ref_min) / (ref_max - ref_min)

def hybrid_score(
    lr_scores,
    iso_scores,
    w_lr: float = 0.8,
    w_iso: float = 0.2,
    ref_stats: dict | None = None,
):
    """
    Combine LR + IsolationForest scores into a single hybrid score.

    ref_stats (optional) should contain:
      {
        "lr_min": ...,
        "lr_max": ...,
        "iso_min": ...,
        "iso_max": ...
      }
    """
    if abs(w_lr + w_iso - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.")

    if ref_stats:
        lr_norm = minmax_normalize(lr_scores, ref_stats["lr_min"], ref_stats["lr_max"])
        iso_norm = minmax_normalize(iso_scores, ref_stats["iso_min"], ref_stats["iso_max"])
    else:
        lr_norm = minmax_normalize(lr_scores)
        iso_norm = minmax_normalize(iso_scores)

    return w_lr * lr_norm + w_iso * iso_norm