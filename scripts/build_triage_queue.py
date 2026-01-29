from src.config import Paths, SplitConfig, DataConfig, ModelConfig
from src.data.load import load_creditcard_csv
from src.data.split import time_based_split
from src.features.preprocess import make_preprocessor, split_xy
from src.models.baseline_logreg import make_logreg_pipeline

from src.decisioning.thresholds import make_triage_thresholds
from src.decisioning.policy import build_triage_queue

import pandas as pd

def band_summary(df_with_band: pd.DataFrame, y_true, band_col: str = "decision_band"):
    y_true = pd.Series(y_true).reset_index(drop=True)
    tmp = df_with_band[[band_col, "risk_score"]].copy()
    tmp["is_fraud"] = y_true.values

    summary = (
        tmp.groupby(band_col)
           .agg(
               n=("is_fraud", "size"),
               fraud_rate=("is_fraud", "mean"),
               avg_score=("risk_score", "mean"),
               max_score=("risk_score", "max"),
           )
           .sort_index()
    )
    return summary

def main():
    paths = Paths()
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig()
    split_cfg = SplitConfig()
    model_cfg = ModelConfig()

    df = load_creditcard_csv(paths.data_dir / data_cfg.filename)
    train_df, val_df, test_df = time_based_split(
        df,
        time_col=data_cfg.time_col,
        label_col=data_cfg.label_col,
        train_frac=split_cfg.train_frac,
        val_frac=split_cfg.val_frac,
        test_frac=split_cfg.test_frac,
    )

    X_train, y_train = split_xy(train_df, data_cfg.label_col)
    X_val, y_val = split_xy(val_df, data_cfg.label_col)
    X_test, y_test = split_xy(test_df, data_cfg.label_col)

    feature_cols = list(X_train.columns)
    scale_cols = [c for c in ["Time", "Amount"] if c in feature_cols]
    preprocessor = make_preprocessor(feature_cols=feature_cols, scale_cols=scale_cols)

    lr = make_logreg_pipeline(preprocessor)
    lr.fit(X_train, y_train)

    val_scores = lr.predict_proba(X_val)[:, 1]
    test_scores = lr.predict_proba(X_test)[:, 1]

    # Choose triage policy (tune these later)
    triage = make_triage_thresholds(val_scores, review_frac=0.005, decline_frac=0.001)
    t_review, t_decline = triage["t_review"], triage["t_decline"]

    triage_test = build_triage_queue(X_test, test_scores, t_review=t_review, t_decline=t_decline)

    summary = band_summary(triage_test, y_test)

    report_path = paths.reports_dir / "triage_preview.md"
    with open(report_path, "w") as f:
        f.write("# Triage Preview (Logistic Regression)\n\n")
        f.write("Policy (chosen on validation scores):\n\n")
        f.write(f"- Review capacity (total flagged): **top {triage['review_frac']*100:.2f}%**\n")
        f.write(f"- Auto-decline capacity: **top {triage['decline_frac']*100:.2f}%**\n")
        f.write(f"- t_review (flag threshold): `{t_review:.6f}`\n")
        f.write(f"- t_decline (decline threshold): `{t_decline:.6f}`\n\n")

        f.write("## Band summary on TEST\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")
        f.write("## Notes\n")
        f.write("- `approve` = below t_review\n")
        f.write("- `review` = between t_review and t_decline\n")
        f.write("- `decline` = above t_decline\n")

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()