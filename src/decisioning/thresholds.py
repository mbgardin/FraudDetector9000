import numpy as np

def threshold_for_top_k(y_score, k_frac: float) -> float:
    """
    Returns the score threshold such that approximately the top k_frac are >= threshold.
    """
    if not (0 < k_frac < 1):
        raise ValueError("k_frac must be in (0, 1)")
    y_score = np.asarray(y_score)
    return float(np.quantile(y_score, 1 - k_frac))

def make_triage_thresholds(
    val_scores,
    review_frac: float = 0.005,
    decline_frac: float = 0.001,
):
    """
    Computes thresholds from validation scores.

    review_frac: fraction of transactions to send to human review OR decline (total flagged)
    decline_frac: fraction of transactions to auto-decline (subset of flagged)

    Must satisfy: 0 < decline_frac < review_frac < 1

    Returns:
      dict with:
        - t_review: score threshold for being flagged at all (review+decline)
        - t_decline: score threshold for auto-decline (highest risk band)
    """
    if not (0 < decline_frac < review_frac < 1):
        raise ValueError("Must satisfy 0 < decline_frac < review_frac < 1")

    t_review = threshold_for_top_k(val_scores, k_frac=review_frac)
    t_decline = threshold_for_top_k(val_scores, k_frac=decline_frac)

    # sanity: decline threshold should be >= review threshold (stricter)
    if t_decline < t_review:
        raise RuntimeError("t_decline should be >= t_review; check inputs/scores.")

    return {
        "t_review": float(t_review),
        "t_decline": float(t_decline),
        "review_frac": float(review_frac),
        "decline_frac": float(decline_frac),
    }