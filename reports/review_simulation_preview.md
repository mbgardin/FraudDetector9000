# Review Simulation Preview

- Loaded triage queue: `triage_queue_test.csv`
- Appended 119 simulated reviews to: `reviews.csv`

## Reviewer action counts (all stored reviews)

| reviewer_action   |   count |
|:------------------|--------:|
| false_positive    |     140 |
| confirmed_fraud   |      16 |
| needs_more_info   |      13 |

## Notes
- This is a prototype simulation to prove the human-in-the-loop data contract.
- In production, reviewer actions arrive asynchronously and may disagree with ground truth.
