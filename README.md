# Fraud Detection / Anomaly Detection System (Credit Card Transactions)

End-to-end fraud risk scoring and human-in-the-loop review workflow.

## What this repo includes
- Supervised + anomaly detection models for transaction fraud
- Thresholding and triage logic (approve / review / decline)
- Evaluation focused on PR-AUC, precision/recall tradeoffs, and cost-aware metrics
- A Streamlit demo that shows flagged transactions with explanations and supports reviewer feedback

## Docs
- docs/problem_statement.md
- docs/dataset.md
- docs/success_metrics.md
- docs/system_scope.md
- docs/decision_policy.md (optional)

## Roadmap
1) Planning docs (current)
2) Baseline model + evaluation
3) Anomaly models + evaluation
4) Ensemble + calibration + decision policy
5) Streamlit demo + human review loop + explanations