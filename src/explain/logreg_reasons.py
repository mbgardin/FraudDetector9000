import numpy as np
import pandas as pd

def _get_feature_names_from_preprocessor(preprocessor, original_columns):
    """
    After ColumnTransformer, order is:
      - transformed columns for the 'scale' transformer (same names as scale_cols)
      - then remainder passthrough columns (in original order, excluding scale_cols)
    We set verbose_feature_names_out=False, but we still need the exact ordering.
    """
    # ColumnTransformer stores transformers_ after fitting.
    # We assume one transformer named "scale" and remainder="passthrough".
    scale_cols = None
    for name, trans, cols in preprocessor.transformers_:
        if name == "scale":
            scale_cols = list(cols)
            break
    if scale_cols is None:
        scale_cols = []

    passthrough_cols = [c for c in original_columns if c not in scale_cols]
    return scale_cols + passthrough_cols

def compute_logreg_contributions(lr_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe of per-feature contributions for each row:
      contribution = coef * transformed_value
    """
    pre = lr_pipeline.named_steps["preprocess"]
    model = lr_pipeline.named_steps["model"]

    Xt = pre.transform(X)
    # coefs shape: (1, n_features)
    coefs = model.coef_.reshape(-1)

    contrib = Xt * coefs  # broadcasting: (n_rows, n_features)
    feature_names = _get_feature_names_from_preprocessor(pre, list(X.columns))
    return pd.DataFrame(contrib, columns=feature_names, index=X.index)

def top_reason_codes(lr_pipeline, X: pd.DataFrame, top_k: int = 3, include_values: bool = False) -> pd.DataFrame:
    """
    For each row, return the top_k positive contributions as reason codes.
    If include_values=True, also include the numeric contribution values.
    """
    contrib_df = compute_logreg_contributions(lr_pipeline, X)

    reasons = []
    values = []
    for _, row in contrib_df.iterrows():
        # Only consider positive contributions (push toward fraud)
        pos = row[row > 0].sort_values(ascending=False)
        top = list(pos.index[:top_k])
        top_vals = list(pos.values[:top_k])

        # Pad if fewer than top_k
        while len(top) < top_k:
            top.append(None)
            top_vals.append(None)

        reasons.append(top)
        values.append(top_vals)

    out = pd.DataFrame(
        reasons,
        columns=[f"reason_{i+1}" for i in range(top_k)],
        index=X.index
    )

    if include_values:
        val_df = pd.DataFrame(
            values,
            columns=[f"reason_{i+1}_score" for i in range(top_k)],
            index=X.index
        )
        out = pd.concat([out, val_df], axis=1)

    return out