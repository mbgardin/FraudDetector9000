from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from .schema import REVIEW_COLUMNS

def init_review_store(path: Path) -> None:
    if not path.exists():
        pd.DataFrame(columns=REVIEW_COLUMNS).to_csv(path, index=False)

def append_reviews(path: Path, rows: list[dict]) -> None:
    init_review_store(path)
    df = pd.read_csv(path)

    ts = datetime.now(timezone.utc).isoformat()
    out_rows = []
    for r in rows:
        rr = r.copy()
        rr["review_timestamp_utc"] = ts
        out_rows.append(rr)

    df2 = pd.concat([df, pd.DataFrame(out_rows)], ignore_index=True)
    df2.to_csv(path, index=False)