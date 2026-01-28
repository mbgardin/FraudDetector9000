from pathlib import Path
import numpy as np

from src.config import Paths, SplitConfig, DataConfig
from src.data.load import load_creditcard_csv
from src.data.split import time_based_split
from src.features.preprocess import make_preprocessor, split_xy
from src.models.baseline_logreg import make_logreg_pipeline
from src.evaluation.metrics import pr_auc, eval_at_threshold, threshold_for_top_k

def main():
    paths = Paths()
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig()
    split_cfg = SplitConfig()

    df = load_creditcard_csv(paths.data_dir / data_cfg.filename)

    train_df, val_df, test_df = time_based_split(
        df,
        time_col=data_cfg.time_col,
        label_col=data_cfg.label_col,
        train_frac=split_cfg.train_frac,
        val_frac=split_cfg.val_frac,
        test_frac=split_cfg.test_frac
    )

    X_train, y_train = split_xy(train_df, data_cfg.label_col)
    X_val, y_val = split_xy(val_df, data_cfg.label_col)
    X_test, y_test = split_xy(test_df, data_cfg.label_col)

    feature_cols = list(X_train.columns)
    scale_cols = [c for c in ["Time", "Amount"] if c in feature_cols]

    preprocessor = make_preprocessor(feature_cols=feature_cols, scale_cols=scale_cols)
    model = make_logreg_pipeline(preprocessor)
    model.fit(X_train, y_train)

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    val_prauc = pr_auc(y_val, val_score)
    test_prauc = pr_auc(y_test, test_score)

    # Thresholds:
    # 1) default
    t_default = 0.5
    # 2) review capacity: top 0.5% flagged
    t_top = threshold_for_top_k(val_score, k_frac=0.005)
    # 3) target recall: pick smallest threshold achieving >= 0.90 recall on val
    candidate_ts = np.linspace(0, 1, 501)
    best_t = 0.5
    for t in candidate_ts:
        stats = eval_at_threshold(y_val.values, val_score, t)
        if stats["recall"] >= 0.90:
            best_t = float(t)
            break

    val_eval = [
        eval_at_threshold(y_val.values, val_score, t_default),
        eval_at_threshold(y_val.values, val_score, t_top),
        eval_at_threshold(y_val.values, val_score, best_t),
    ]
    test_eval = [
        eval_at_threshold(y_test.values, test_score, t_default),
        eval_at_threshold(y_test.values, test_score, t_top),
        eval_at_threshold(y_test.values, test_score, best_t),
    ]

    report_path = paths.reports_dir / "baseline.md"
    with open(report_path, "w") as f:
        f.write("# Baseline Report — Logistic Regression\n\n")
        f.write(f"- Validation PR-AUC: **{val_prauc:.4f}**\n")
        f.write(f"- Test PR-AUC: **{test_prauc:.4f}**\n\n")

        f.write("## Threshold evaluations (Validation)\n")
        for row in val_eval:
            f.write(f"\n### Threshold = {row['threshold']:.4f}\n")
            f.write(f"- Precision: {row['precision']:.4f}\n")
            f.write(f"- Recall: {row['recall']:.4f}\n")
            f.write(f"- F1: {row['f1']:.4f}\n")
            f.write(f"- Confusion matrix: {row['confusion_matrix']}\n")

        f.write("\n## Threshold evaluations (Test)\n")
        for row in test_eval:
            f.write(f"\n### Threshold = {row['threshold']:.4f}\n")
            f.write(f"- Precision: {row['precision']:.4f}\n")
            f.write(f"- Recall: {row['recall']:.4f}\n")
            f.write(f"- F1: {row['f1']:.4f}\n")
            f.write(f"- Confusion matrix: {row['confusion_matrix']}\n")

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()