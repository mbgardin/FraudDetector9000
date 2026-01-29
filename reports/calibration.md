# Probability Calibration Report (Logistic Regression)

Base model trained on TRAIN. Calibrators fit on VAL. Metrics reported on TEST.

## Metrics (TEST)

| Variant | PR-AUC | Log Loss | Brier |
|---|---:|---:|---:|
| raw | 0.7099 | 0.069988 | 0.015042 |
| sigmoid | 0.7003 | 0.003356 | 0.000444 |
| isotonic | 0.7254 | 0.003646 | 0.000457 |

## Plots

- Sigmoid calibration curve saved to: `calibration_curve_sigmoid.png`
- Isotonic calibration curve saved to: `calibration_curve_isotonic.png`

## Notes
- PR-AUC mostly reflects ranking; calibration focuses on probability correctness.
- Log loss and Brier score should typically improve after calibration.
- Calibrated probabilities are more defensible for human-facing triage thresholds.
