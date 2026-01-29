# Triage Preview (Logistic Regression)

Policy (chosen on validation scores):

- Review capacity (total flagged): **top 0.50%**
- Auto-decline capacity: **top 0.10%**
- t_review (flag threshold): `0.874479`
- t_decline (decline threshold): `0.999998`

## Band summary on TEST

| decision_band   |     n |   fraud_rate |   avg_score |   max_score |
|:----------------|------:|-------------:|------------:|------------:|
| approve         | 42569 |  0.000258404 |   0.0478805 |    0.873981 |
| decline         |    34 |  0.941176    |   1         |    1        |
| review          |   119 |  0.0756303   |   0.948722  |    0.999997 |

## Notes
- `approve` = below t_review
- `review` = between t_review and t_decline
- `decline` = above t_decline
