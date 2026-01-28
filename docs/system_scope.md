# System Scope (MVP → Stretch)

## MVP scope (what must exist)
- A clean baseline supervised model + evaluation
- At least one anomaly model + evaluation
- A scoring function that outputs:
  - risk score
  - decision band (approve/review/decline)
  - explanation summary
- Streamlit demo with a review queue and feedback capture

## Stretch scope (nice-to-have)
- Autoencoder anomaly detection
- Stacking / ensemble meta-model
- Segment-specific thresholds
- Drift monitoring visuals
- Feedback loop simulation (retrain using reviewer labels)
- More realistic dataset extension (IEEE-CIS)