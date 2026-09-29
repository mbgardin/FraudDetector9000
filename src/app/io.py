from pathlib import Path
import pandas as pd

import numpy as np

def generate_fallback_queue() -> pd.DataFrame:
    """Generate a realistic fallback triage queue if data file is missing."""
    np.random.seed(42)
    n_rows = 300
    indices = np.arange(242000, 242000 + n_rows)
    
    # Generate scores across bands
    decline_scores = np.random.uniform(0.75, 0.99, size=25)
    review_scores = np.random.uniform(0.35, 0.74, size=75)
    approve_scores = np.random.uniform(0.001, 0.34, size=200)
    
    scores = np.concatenate([decline_scores, review_scores, approve_scores])
    bands = ["decline"] * 25 + ["review"] * 75 + ["approve"] * 200
    
    sample_reasons = [
        ("V14_extreme_negative", "Amount_high_anomaly", "V17_negative_signal"),
        ("V12_anomaly_score", "V10_suspicious_pattern", "Time_off_peak"),
        ("V4_positive_shift", "V11_spike", "Amount_unusual"),
        ("V14_extreme_negative", "V4_positive_shift", "V3_deviation"),
    ]
    
    r1, r2, r3 = [], [], []
    for b in bands:
        if b in ("review", "decline"):
            choice = sample_reasons[np.random.randint(len(sample_reasons))]
            r1.append(choice[0])
            r2.append(choice[1])
            r3.append(choice[2])
        else:
            r1.append(None)
            r2.append(None)
            r3.append(None)
            
    df = pd.DataFrame({
        "transaction_index": indices,
        "risk_score": scores,
        "decision_band": bands,
        "reason_1": r1,
        "reason_2": r2,
        "reason_3": r3,
        "Amount": np.random.exponential(scale=50.0, size=n_rows).round(2),
        "Time": np.random.uniform(100000, 150000, size=n_rows).round(0),
    })
    return df

def load_triage_queue(path: Path) -> pd.DataFrame:
    if not path.exists():
        df = generate_fallback_queue()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        except Exception:
            pass
        return df

    df = pd.read_csv(path)

    # normalize transaction index
    if "transaction_index" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "transaction_index"})
        else:
            raise ValueError("triage queue must include transaction_index column")

    # --- normalize types and strings (prevents empty tables from filter mismatches) ---
    df["transaction_index"] = df["transaction_index"].astype(int)

    # decision_band sometimes has spaces/case issues
    df["decision_band"] = df["decision_band"].astype(str).str.strip().str.lower()

    # risk_score sometimes loads as object/string
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")

    # drop rows with missing score (should be none, but safe)
    df = df.dropna(subset=["risk_score"])

    return df

def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "transaction_index","risk_score","decision_band",
            "reviewer_action","reviewer_notes","review_timestamp_utc"
        ])
    return pd.read_csv(path)

def reviews_for_transaction(reviews_df: pd.DataFrame, transaction_index: int) -> pd.DataFrame:
    if reviews_df.empty:
        return reviews_df
    return reviews_df[reviews_df["transaction_index"].astype(int) == int(transaction_index)]