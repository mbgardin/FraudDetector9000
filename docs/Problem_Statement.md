# Problem Statement — Credit Card Transaction Fraud Detection

## Context
Banks and payment processors must decide, in near real time, whether to approve or decline card transactions. Fraud is rare, costly, and constantly evolving. A good system must catch as much fraud as possible without creating too many false positives that frustrate customers and overwhelm review teams.

This project builds an end-to-end fraud risk scoring and triage system that:
- scores each transaction for fraud risk,
- recommends an action (approve / review / decline),
- provides a human-readable explanation for every flagged case,
- supports a human review workflow and feedback capture.

## Objective
Given transaction data, output:
1) a fraud risk score,
2) a decision recommendation (approve/review/decline),
3) an explanation (reason codes / top contributing features).

## What counts as “fraud” here?
For evaluation, fraud is the dataset’s binary fraud label. In production, labels are delayed and incomplete (chargebacks, investigations, missed fraud). This repo treats dataset labels as ground truth for modeling, but designs the system with label delay and partial labeling in mind.

## Constraints & assumptions
- Fraud is extremely rare (severe class imbalance).
- False negatives are expensive (missed fraud = direct loss).
- False positives also matter (bad customer experience + operational review load).
- The distribution changes over time (concept drift and adversarial adaptation).
- The system must support human-in-the-loop review.

## Success criteria
We do not optimize for accuracy. We optimize for:
- PR-AUC and precision/recall tradeoffs,
- recall at a controlled false positive rate (or controlled review capacity),
- precision@K (quality of the top reviewed alerts),
- cost-aware evaluation with a simple business cost model,
- stability over time using time-based splits (no random leakage),
- actionability: every flagged transaction includes an explanation.

## Outputs
- Reproducible training + evaluation pipeline
- Model comparison report across multiple approaches (supervised + anomaly)
- Streamlit demo showing:
  - transaction details + score + decision
  - explanation for why it was flagged
  - human review actions (confirm fraud / false positive / needs info)
  - stored reviewer outcomes for feedback simulation