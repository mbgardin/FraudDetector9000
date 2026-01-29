from src.config import Paths, SplitConfig, DataConfig, ModelConfig
from src.data.load import load_creditcard_csv
from src.data.split import time_based_split
from src.features.preprocess import make_preprocessor, split_xy

from src.models.baseline_logreg import make_logreg_pipeline
from src.models.isolation_forest import make_isoforest_pipeline, anomaly_score as iso_score
from src.models.hybrid import hybrid_score

from src.evaluation.metrics import pr_auc, eval_at_threshold, threshold_for_top_k

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

    # ---- Train models ----
    lr = make_logreg_pipeline(preprocessor)
    lr.fit(X_train, y_train)

    iso = make_isoforest_pipeline(preprocessor, random_state=model_cfg.random_state)
    iso.fit(X_train)

    # ---- Scores ----
    lr_val = lr.predict_proba(X_val)[:, 1]
    lr_test = lr.predict_proba(X_test)[:, 1]

    iso_val = iso_score(iso, X_val)
    iso_test = iso_score(iso, X_test)

    # ---- Normalization reference from validation ----
    ref = {
        "lr_min": lr_val.min(),
        "lr_max": lr_val.max(),
        "iso_min": iso_val.min(),
        "iso_max": iso_val.max(),
    }

    hybrid_val = hybrid_score(lr_val, iso_val, w_lr=0.8, w_iso=0.2, ref_stats=ref)
    hybrid_test = hybrid_score(lr_test, iso_test, w_lr=0.8, w_iso=0.2, ref_stats=ref)

    # ---- Compare under same capacity ----
    k_frac = 0.005  # top 0.5%

    rows = []

    for name, vscore, tscore in [
        ("LogisticRegression", lr_val, lr_test),
        ("Hybrid(LR+Iso)", hybrid_val, hybrid_test),
    ]:
        t_thresh = threshold_for_top_k(vscore, k_frac=k_frac)

        v_stats = eval_at_threshold(y_val.values, vscore, t_thresh)
        t_stats = eval_at_threshold(y_test.values, tscore, t_thresh)

        rows.append({
            "model": name,
            "val_pr_auc": pr_auc(y_val, vscore),
            "test_pr_auc": pr_auc(y_test, tscore),
            "test_precision@top0.5%": t_stats["precision"],
            "test_recall@top0.5%": t_stats["recall"],
        })

    report_path = paths.reports_dir / "hybrid_comparison.md"
    with open(report_path, "w") as f:
        f.write("# Hybrid Scoring Comparison\n\n")
        f.write("Hybrid score = 0.8 × LR + 0.2 × IsolationForest (min-max normalized on validation).\n\n")
        f.write("| Model | Val PR-AUC | Test PR-AUC | Test Precision@Top0.5% | Test Recall@Top0.5% |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | {r['val_pr_auc']:.4f} | {r['test_pr_auc']:.4f} "
                f"| {r['test_precision@top0.5%']:.4f} | {r['test_recall@top0.5%']:.4f} |\n"
            )

        f.write("\n## Notes\n")
        f.write("- Hybrid models often trade a bit of recall for better novelty coverage.\n")
        f.write("- Weights are illustrative and can be tuned.\n")

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()