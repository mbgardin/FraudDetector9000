# Problem Statement — Credit Card Transaction Fraud Detection

## Context
Payment networks and banks must decide, in near real time, whether to approve or decline a credit card transaction. Fraud is rare, constantly evolving, and expensive. Declining legitimate transactions frustrates customers and loses revenue, while missing fraud directly costs money and increases downstream operational burden (chargebacks, investigations, reputation damage).

This project builds an end-to-end fraud risk scoring system that flags suspicious credit card transactions and routes them into a human review workflow. The goal is not “maximum accuracy,” but a practical decision pipeline that balances fraud prevention with customer experience.

## Objective
Given a stream of card transactions, produce:
1) a fraud risk score (0–1 or 0–100),
2) a decision recommendation (approve / review / decline),
3) an explanation for why the transaction was flagged.

## What is “Fraud” here?
Fraud is defined as a transaction labeled as fraudulent in the dataset (binary label). In production, fraud labels are typically delayed and incomplete (many fraud cases are never labeled, and some labels are wrong). This repo treats dataset labels as ground truth for evaluation, but designs the system with realistic label issues in mind.

## Constraints & Real-World Assumptions
- **Severe class imbalance:** fraud is a tiny fraction of total transactions.
- **High cost of false negatives:** missed fraud directly costs money.
- **Non-trivial cost of false positives:** declines/reviews harm user experience and create operational load.
- **Time sensitivity:** decisions must be made quickly (milliseconds to seconds).
- **Concept drift:** fraud patterns change over time; monitoring and retraining are required.
- **Human-in-the-loop:** a portion of flagged transactions must be reviewed by analysts.

## Success Criteria (How we measure “good”)
This project prioritizes:
- **Recall at a fixed false positive rate** (or fixed review capacity),
- **Precision-Recall AUC (PR-AUC)** over ROC-AUC,
- **Cost-aware evaluation** using a simple dollar-weighted model,
- **Stability over time** using time-based splits (no random leakage),
- **Actionability**: every flag should include an explanation that a human can understand.

## Outputs
- Batch scoring + evaluation reports
- A Streamlit app (or API) that shows:
  - transaction details,
  - risk score,
  - decision (approve/review/decline),
  - explanation (top contributing features / reason codes),
  - reviewer feedback capture (confirm fraud / false positive / needs more info)

## Dataset
We start with the public “Credit Card Fraud Detection” dataset (European cardholders, 2 days, 284,807 transactions, 492 fraud cases). Features are anonymized (PCA components V1–V28) plus Time and Amount. This dataset is used to prototype the system and demonstrate end-to-end design patterns.