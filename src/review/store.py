import pandas as pd
from pathlib import Path
from datetime import datetime
from .schema import REVIEW_COLUMNS

def init_review_store(path: Path):
    if not path.exists():
        df = pd.DataFrame(columns=REVIEW_COLUMNS)
        df.to_csv(path, index=False)

def append_reviews(path: Path, rows: list[dict]):
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REVIEW_COLUMNS)

    for r in rows:
        r = r.copy()
        r["review_timestamp"] = datetime.utcnow().isoformat()
        df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)

    df.to_csv(path, index=False)