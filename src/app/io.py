from pathlib import Path
import pandas as pd

def load_triage_queue(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing triage queue at {path}. Run: python -m scripts.build_triage_queue_with_reasons"
        )
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