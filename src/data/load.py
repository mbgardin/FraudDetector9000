import pandas as pd
from pathlib import Path

def load_creditcard_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {path}. Put creditcard.csv in /data."
        )
    df = pd.read_csv(path)
    return df