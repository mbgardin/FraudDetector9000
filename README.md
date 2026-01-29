# Fraud Detection & Human-in-the-Loop Review System

## Overview

This project implements an **end-to-end fraud detection system** designed to mirror real-world production workflows.  
Rather than focusing only on model accuracy, the system emphasizes **decision-making under operational constraints**, **explainability**, and **human analyst interaction**.

The system scores transactions, assigns them to decision bands (approve / review / decline), provides per-transaction explanations, and supports a full **human-in-the-loop review workflow** with persistent feedback storage.

---

## Key Features

- Time-aware train / validation / test split (prevents leakage)
- Supervised + anomaly-based modeling
- Evaluation under fixed review capacity (not just accuracy)
- Probability calibration for human-facing decisions
- Per-transaction reason codes
- Human review queue with audit trail
- Interactive Streamlit demo

---

## System Architecture

### 1. Data Handling
- Credit card transaction data with extreme class imbalance
- Time-based splitting to reflect real deployment scenarios

### 2. Risk Modeling
- **Logistic Regression**
  - Primary supervised risk signal
  - Optimized for precision/recall tradeoffs under review capacity
- **Isolation Forest**
  - Captures novel or unusual transaction behavior
- **Hybrid Scoring (optional)**
  - Weighted blend of supervised + anomaly signals

### 3. Calibration
- Sigmoid (Platt scaling) and isotonic calibration
- Calibration fitted on validation data
- Improves probability trustworthiness (log loss & Brier score)

### 4. Decision Policy
Capacity-based triage using validation-derived thresholds:

- **Approve**: low risk, auto-allow  
- **Review**: routed to human analysts  
- **Decline**: highest risk, block or escalate  

This mirrors real fraud operations where analyst capacity is limited.

### 5. Explainability
- Per-transaction reason codes derived from model feature contributions
- Highlights which signals pushed a transaction toward fraud
- Supports analyst trust and auditability

### 6. Human-in-the-Loop Workflow
- Review queue with filters and prioritization
- Analysts submit decisions and notes
- Feedback persisted for auditing and future retraining
- Full review history visible per transaction

---

## Evaluation Philosophy

This project intentionally avoids accuracy as a primary metric.

Key evaluation choices:
- **PR-AUC** for extreme class imbalance
- **Precision / Recall at fixed review capacity**
- Shared thresholds across models for fair comparison
- Separate evaluation of ranking quality vs probability quality

Calibration is evaluated using:
- Log loss
- Brier score
- Calibration curves

---

## Streamlit Demo

The Streamlit app demonstrates the full review workflow:

- Filterable review queue
- Transaction-level explanations
- Review submission with notes
- Persistent review history
- Human feedback loop in action

### Run the demo locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
## Author
### Monte Gardiner
Statistics and Data Science
Git hub: mbgardin
LinkedIn: 