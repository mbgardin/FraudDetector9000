import numpy as np
import pandas as pd
from typing import Tuple

def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    label_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # sanity checks
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if label_col not in split_df.columns:
            raise ValueError(f"{label_col} not found in {split_name} split.")
        if time_col not in split_df.columns:
            raise ValueError(f"{time_col} not found in {split_name} split.")

    return train_df, val_df, test_df