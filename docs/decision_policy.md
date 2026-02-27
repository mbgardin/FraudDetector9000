# Decision Policy (Human-in-the-loop)

I do not use a single “fraud/not fraud” decision. I use a triage policy:

## Risk bands
- **Approve**: low risk, no human review
- **Review**: medium risk, send to analyst queue
- **Decline**: high risk, auto-block (or auto-block above stricter threshold)

## Threshold types we will support
- Static thresholds (simple baseline)
- Cost-based thresholds (minimize expected cost)
- Review-capacity thresholds (Top-K per day)
- Segment-specific thresholds (amount buckets, new vs returning, etc.)
- Two-stage gating: cheap model first, expensive model second for borderline cases

## Human review workflow
Every reviewed alert must show:
- model score + band
- explanation / reason codes
- reviewer outcome:
  - confirmed fraud
  - false positive
  - needs more info

Reviewer outcomes are stored and used to simulate continuous improvement.