import numpy as np
import pandas as pd

def assign_decision_band(scores, t_review: float, t_decline: float):
    """
    Map numeric scores to decision bands:
      - approve: score < t_review
      - review:  t_review <= score < t_decline
      - decline: score >= t_decline
    """
    s = np.asarray(scores, dtype=float)

    decision = np.full(shape=s.shape, fill_value="approve", dtype=object)
    decision[s >= t_review] = "review"
    decision[s >= t_decline] = "decline"
    return decision

def build_triage_queue(
    df_features: pd.DataFrame,
    scores,
    t_review: float,
    t_decline: float,
    id_col: str = None,
):
    """
    Returns a dataframe with id (optional), score, decision_band,
    and the original features (for downstream explanation + UI).
    """
    out = df_features.copy()
    out["risk_score"] = np.asarray(scores, dtype=float)
    out["decision_band"] = assign_decision_band(out["risk_score"].values, t_review, t_decline)

    if id_col and id_col in out.columns:
        cols = [id_col, "risk_score", "decision_band"] + [c for c in out.columns if c not in {id_col, "risk_score", "decision_band"}]
        out = out[cols]
    else:
        cols = ["risk_score", "decision_band"] + [c for c in out.columns if c not in {"risk_score", "decision_band"}]
        out = out[cols]

    return out