from src.config import Paths, SplitConfig, DataConfig, ModelConfig
from src.data.load import load_creditcard_csv
from src.data.split import time_based_split
from src.features.preprocess import make_preprocessor, split_xy

from src.models.baseline_logreg import make_logreg_pipeline
from src.models.isolation_forest import make_isoforest_pipeline, anomaly_score as iso_score
from src.models.lof import make_lof_pipeline, anomaly_score as lof_score

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

    # ---- Model 1: Logistic Regression ----
    lr = make_logreg_pipeline(preprocessor)
    lr.fit(X_train, y_train)
    lr_val = lr.predict_proba(X_val)[:, 1]
    lr_test = lr.predict_proba(X_test)[:, 1]

    # ---- Model 2: Isolation Forest ----
    iso = make_isoforest_pipeline(preprocessor, random_state=model_cfg.random_state)
    iso.fit(X_train)
    iso_val = iso_score(iso, X_val)
    iso_test = iso_score(iso, X_test)

    # ---- Model 3: LOF ----
    lof = make_lof_pipeline(preprocessor, n_neighbors=35)
    lof.fit(X_train)
    lof_val = lof_score(lof, X_val)
    lof_test = lof_score(lof, X_test)

    models = [
        ("LogisticRegression", lr_val, lr_test),
        ("IsolationForest", iso_val, iso_test),
        ("LOF", lof_val, lof_test),
    ]

    # Compare at a shared "review capacity": top 0.5% flagged using VAL threshold
    k_frac = 0.005

    rows = []
    for name, vscore, tscore in models:
        v_prauc = pr_auc(y_val, vscore)
        t_prauc = pr_auc(y_test, tscore)

        t_thresh = threshold_for_top_k(vscore, k_frac=k_frac)

        v_stats = eval_at_threshold(y_val.values, vscore, t_thresh)
        t_stats = eval_at_threshold(y_test.values, tscore, t_thresh)

        rows.append({
            "model": name,
            "val_pr_auc": v_prauc,
            "test_pr_auc": t_prauc,
            "val_precision@top0.5%": v_stats["precision"],
            "val_recall@top0.5%": v_stats["recall"],
            "test_precision@top0.5%": t_stats["precision"],
            "test_recall@top0.5%": t_stats["recall"],
        })

    report_path = paths.reports_dir / "model_comparison.md"
    with open(report_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"Shared evaluation policy: **flag top {k_frac*100:.2f}%** by score using a validation-set threshold.\n\n")
        f.write("| Model | Val PR-AUC | Test PR-AUC | Val Precision@Top0.5% | Val Recall@Top0.5% | Test Precision@Top0.5% | Test Recall@Top0.5% |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['model']} "
                f"| {r['val_pr_auc']:.4f} | {r['test_pr_auc']:.4f} "
                f"| {r['val_precision@top0.5%']:.4f} | {r['val_recall@top0.5%']:.4f} "
                f"| {r['test_precision@top0.5%']:.4f} | {r['test_recall@top0.5%']:.4f} |\n"
            )

        f.write("\n## Notes\n")
        f.write("- PR-AUC is the primary metric due to extreme class imbalance.\n")
        f.write("- IsolationForest and LOF are evaluated as anomaly scorers (higher score = more anomalous).\n")
        f.write("- LOF may perform near base rate on this dataset; this is still a valuable result because it shows proper evaluation and model selection.\n")

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()