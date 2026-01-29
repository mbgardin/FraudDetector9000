from src.config import Paths, SplitConfig, DataConfig, ModelConfig
from src.data.load import load_creditcard_csv
from src.data.split import time_based_split
from src.features.preprocess import make_preprocessor, split_xy
from src.models.baseline_logreg import make_logreg_pipeline

from src.decisioning.thresholds import make_triage_thresholds
from src.decisioning.policy import build_triage_queue
from src.explain import top_reason_codes

import pandas as pd

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

    triage = make_triage_thresholds(val_scores, review_frac=0.005, decline_frac=0.001)
    t_review, t_decline = triage["t_review"], triage["t_decline"]

    triage_test = build_triage_queue(X_test, test_scores, t_review=t_review, t_decline=t_decline)

    # Add reason codes only for flagged rows (review/decline)
    flagged_mask = triage_test["decision_band"].isin(["review", "decline"])
    flagged_X = X_test.loc[flagged_mask].copy()

    reasons = top_reason_codes(lr, flagged_X, top_k=3, include_values=True)

    triage_test = triage_test.join(reasons, how="left")

    # Preview: show a few flagged rows (with label for sanity)
    preview = triage_test.loc[flagged_mask, ["risk_score", "decision_band", "reason_1", "reason_2", "reason_3",
                                            "reason_1_score", "reason_2_score", "reason_3_score"]].copy()
    preview["is_fraud"] = pd.Series(y_test.values, index=X_test.index).loc[flagged_mask].values
    preview = preview.sort_values(["decision_band", "risk_score"], ascending=[True, False]).head(25)

    report_path = paths.reports_dir / "triage_with_reasons_preview.md"
    with open(report_path, "w") as f:
        f.write("# Triage Preview with Reason Codes (Logistic Regression)\n\n")
        f.write("Top 25 flagged items on TEST (sorted by band then score).\n\n")
        f.write(preview.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Notes\n")
        f.write("- Reasons are the top positive logreg feature contributions for that row.\n")
        f.write("- Contribution = coefficient × (transformed feature value).\n")
        f.write("- Features are anonymized (V1–V28), so reasons are best interpreted as *signals* rather than human-meaningful fields.\n")

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()