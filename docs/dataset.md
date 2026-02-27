# Dataset

## Primary dataset
I started with the public "Credit Card Fraud Detection" dataset (often referenced as the MLG-ULB dataset).

High-level characteristics:
- ~285k transactions
- ~492 labeled fraud cases (very imbalanced)
- Features are anonymized (PCA-transformed) plus `Time` and `Amount`
- Contains a binary label `Class` (1 = fraud, 0 = normal)

## Why this dataset
- Forces imbalanced learning and realistic metrics (PR curves, recall/precision)
- Fast iteration cycle for building an end-to-end system
- Good sandbox for thresholding, triage logic, and human review workflow

## Known limitations (we design around them)
- Features are anonymized (harder to interpret than real banking features)
- Short time window (limits long-term drift analysis)
- Lacks entity IDs (card_id / merchant_id), so some “velocity” features are not possible unless simulated

## Planned extension dataset (optional)
After the MVP, we can extend to a more realistic dataset (e.g., IEEE-CIS Fraud Detection) to demonstrate messier joins, identity features, and more realistic leakage concerns.