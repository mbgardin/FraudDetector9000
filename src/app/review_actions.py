from pathlib import Path
import pandas as pd
from src.review import init_review_store, append_reviews

def mark_reviewed(review_path: Path, transaction_index: int, risk_score: float, decision_band: str,
                  reviewer_action: str, reviewer_notes: str):
    init_review_store(review_path)
    append_reviews(review_path, [{
        "transaction_index": int(transaction_index),
        "risk_score": float(risk_score),
        "decision_band": str(decision_band),
        "reviewer_action": str(reviewer_action),
        "reviewer_notes": str(reviewer_notes),
    }])

def reviewed_set(reviews_df: pd.DataFrame) -> set[int]:
    if reviews_df.empty:
        return set()
    return set(reviews_df["transaction_index"].astype(int).tolist())