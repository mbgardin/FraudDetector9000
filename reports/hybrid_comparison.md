# Hybrid Scoring Comparison

Hybrid score = 0.8 × LR + 0.2 × IsolationForest (min-max normalized on validation).

| Model | Val PR-AUC | Test PR-AUC | Test Precision@Top0.5% | Test Recall@Top0.5% |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.8401 | 0.7099 | 0.2680 | 0.7885 |
| Hybrid(LR+Iso) | 0.3531 | 0.4076 | 0.2733 | 0.7885 |

## Notes
- Hybrid models often trade a bit of recall for better novelty coverage.
- Weights are illustrative and can be tuned.
