import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def make_preprocessor(feature_cols, scale_cols):
    # Scale only selected cols, pass through the rest
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

def split_xy(df: pd.DataFrame, label_col: str):
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)
    return X, y