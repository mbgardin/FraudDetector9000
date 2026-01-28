# Success Metrics & Evaluation Plan

## Primary metrics (what we care about)
Fraud detection is imbalanced, so accuracy is not meaningful. Primary metrics:
- **PR-AUC** (Precision-Recall Area Under Curve)
- **Recall at a fixed FPR** (or at a fixed review capacity)
- **Precision@K** (quality of top K alerts sent to human review)
- **Confusion matrix at multiple thresholds** (approve/review/decline policy)
- **Calibration** (probabilities should mean something if we show them to humans)

## Cost-aware evaluation (simple business model)
We evaluate decisions with a simple cost model:
- False Negative cost (missed fraud) = high
- False Positive cost (wrong decline/review) = moderate
- True Positive benefit = prevented loss
- Review cost = analyst time

We will report cost vs threshold curves to show the tradeoff.

## Splitting strategy (avoid leakage)
We prefer time-based splits when possible:
- Train on earlier time, test on later time
- No random split that mixes “future” patterns into training

If the dataset’s time field is limited, we still enforce:
- no preprocessing fit on test data (scalers, PCA already provided, etc.)
- thresholds chosen on validation set only