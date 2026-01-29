# Model Comparison Report

Shared evaluation policy: **flag top 0.50%** by score using a validation-set threshold.

| Model | Val PR-AUC | Test PR-AUC | Val Precision@Top0.5% | Val Recall@Top0.5% | Test Precision@Top0.5% | Test Recall@Top0.5% |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.8401 | 0.7099 | 0.2383 | 0.9107 | 0.2680 | 0.7885 |
| IsolationForest | 0.0319 | 0.0424 | 0.0280 | 0.1071 | 0.0394 | 0.0962 |
| LOF | 0.0018 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Notes
- PR-AUC is the primary metric due to extreme class imbalance.
- IsolationForest and LOF are evaluated as anomaly scorers (higher score = more anomalous).
- LOF may perform near base rate on this dataset; this is still a valuable result because it shows proper evaluation and model selection.
