# Baseline Report — Logistic Regression

- Validation PR-AUC: **0.8401**
- Test PR-AUC: **0.7099**

## Threshold evaluations (Validation)

### Threshold = 0.5000
- Precision: 0.0528
- Recall: 0.9286
- F1: 0.1000
- Confusion matrix: [[41733, 932], [4, 52]]

### Threshold = 0.8745
- Precision: 0.2383
- Recall: 0.9107
- F1: 0.3778
- Confusion matrix: [[42502, 163], [5, 51]]

### Threshold = 0.0000
- Precision: 0.0013
- Recall: 1.0000
- F1: 0.0026
- Confusion matrix: [[0, 42665], [0, 56]]

## Threshold evaluations (Test)

### Threshold = 0.5000
- Precision: 0.0568
- Recall: 0.8269
- F1: 0.1063
- Confusion matrix: [[41956, 714], [9, 43]]

### Threshold = 0.8745
- Precision: 0.2680
- Recall: 0.7885
- F1: 0.4000
- Confusion matrix: [[42558, 112], [11, 41]]

### Threshold = 0.0000
- Precision: 0.0012
- Recall: 1.0000
- F1: 0.0024
- Confusion matrix: [[0, 42670], [0, 52]]
