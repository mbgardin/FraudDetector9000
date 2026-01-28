import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix

def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))

def eval_at_threshold(y_true, y_score, thresh: float):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    y_pred = (y_score >= thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(thresh),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

def threshold_for_top_k(y_score, k_frac: float) -> float:
    assert 0 < k_frac < 1
    y_score = np.asarray(y_score)
    return float(np.quantile(y_score, 1 - k_frac))