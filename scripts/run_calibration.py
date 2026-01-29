from src.config import Paths, SplitConfig, DataConfig
from src.data.load import load_creditcard_csv
from src.data.split import time_based_split
from src.features.preprocess import make_preprocessor, split_xy
from src.models.baseline_logreg import make_logreg_pipeline
from src.calibration import fit_calibrator_prefit, get_calibration_curve, brier_score

from sklearn.metrics import log_loss, average_precision_score
import matplotlib.pyplot as plt

def plot_calibration(frac_pos, mean_pred, out_path):
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

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
        test_frac=split_cfg.test_frac,
    )

    X_train, y_train = split_xy(train_df, data_cfg.label_col)
    X_val, y_val = split_xy(val_df, data_cfg.label_col)
    X_test, y_test = split_xy(test_df, data_cfg.label_col)

    feature_cols = list(X_train.columns)
    scale_cols = [c for c in ["Time", "Amount"] if c in feature_cols]
    preprocessor = make_preprocessor(feature_cols=feature_cols, scale_cols=scale_cols)

    # ---- base model ----
    lr = make_logreg_pipeline(preprocessor)
    lr.fit(X_train, y_train)

    p_val_raw = lr.predict_proba(X_val)[:, 1]
    p_test_raw = lr.predict_proba(X_test)[:, 1]

    # ---- calibrators fit on validation ----
    cal_sigmoid = fit_calibrator_prefit(lr, X_val, y_val, method="sigmoid")
    cal_isotonic = fit_calibrator_prefit(lr, X_val, y_val, method="isotonic")

    p_test_sigmoid = cal_sigmoid.predict_proba(X_test)[:, 1]
    p_test_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]

    # ---- metrics ----
    metrics = {
        "raw": {
            "pr_auc": float(average_precision_score(y_test, p_test_raw)),
            "log_loss": float(log_loss(y_test, p_test_raw, labels=[0, 1])),
            "brier": float(brier_score(y_test, p_test_raw)),
        },
        "sigmoid": {
            "pr_auc": float(average_precision_score(y_test, p_test_sigmoid)),
            "log_loss": float(log_loss(y_test, p_test_sigmoid, labels=[0, 1])),
            "brier": float(brier_score(y_test, p_test_sigmoid)),
        },
        "isotonic": {
            "pr_auc": float(average_precision_score(y_test, p_test_isotonic)),
            "log_loss": float(log_loss(y_test, p_test_isotonic, labels=[0, 1])),
            "brier": float(brier_score(y_test, p_test_isotonic)),
        },
    }

    # ---- calibration curves ----
    frac_raw, mean_raw = get_calibration_curve(y_test, p_test_raw, n_bins=10)
    frac_sig, mean_sig = get_calibration_curve(y_test, p_test_sigmoid, n_bins=10)
    frac_iso, mean_iso = get_calibration_curve(y_test, p_test_isotonic, n_bins=10)

    out_sig = paths.reports_dir / "calibration_curve_sigmoid.png"
    out_iso = paths.reports_dir / "calibration_curve_isotonic.png"
    plot_calibration(frac_sig, mean_sig, out_sig)
    plot_calibration(frac_iso, mean_iso, out_iso)

    # ---- report ----
    report_path = paths.reports_dir / "calibration.md"
    with open(report_path, "w") as f:
        f.write("# Probability Calibration Report (Logistic Regression)\n\n")
        f.write("Base model trained on TRAIN. Calibrators fit on VAL. Metrics reported on TEST.\n\n")

        f.write("## Metrics (TEST)\n\n")
        f.write("| Variant | PR-AUC | Log Loss | Brier |\n")
        f.write("|---|---:|---:|---:|\n")
        for k in ["raw", "sigmoid", "isotonic"]:
            f.write(
                f"| {k} | {metrics[k]['pr_auc']:.4f} | {metrics[k]['log_loss']:.6f} | {metrics[k]['brier']:.6f} |\n"
            )

        f.write("\n## Plots\n\n")
        f.write(f"- Sigmoid calibration curve saved to: `{out_sig.name}`\n")
        f.write(f"- Isotonic calibration curve saved to: `{out_iso.name}`\n\n")

        f.write("## Notes\n")
        f.write("- PR-AUC mostly reflects ranking; calibration focuses on probability correctness.\n")
        f.write("- Log loss and Brier score should typically improve after calibration.\n")
        f.write("- Calibrated probabilities are more defensible for human-facing triage thresholds.\n")

    print(f"Wrote {report_path}")
    print(f"Wrote {out_sig}")
    print(f"Wrote {out_iso}")

if __name__ == "__main__":
    main()