from src.config import Paths
from src.review import init_review_store, append_reviews

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    paths = Paths()
    review_path = paths.data_dir / "reviews.csv"

    init_review_store(review_path)

    # Load the triage-with-reasons preview as a proxy for the queue
    triage_path = paths.reports_dir / "triage_with_reasons_preview.md"

    # Instead of parsing markdown, reload data fresh (simpler & realistic)
    # In real systems, this would come from a DB or service.
    print("Simulating review decisions using ground truth...")

    # For simulation, regenerate triage data via CSV-like approach
    # We assume the user runs this after Step 4B, so we mock rows:
    # (In practice, you'd persist the triage queue as a CSV or DB table.)

    # For now: simulate 50 reviews
    rng = np.random.default_rng(42)

    simulated_rows = []
    for i in range(50):
        simulated_rows.append({
            "transaction_index": i,
            "risk_score": rng.uniform(0.8, 1.0),
            "decision_band": "review",
            "reviewer_action": rng.choice(
                ["confirmed_fraud", "false_positive", "needs_more_info"],
                p=[0.15, 0.75, 0.10]
            ),
            "reviewer_notes": "Simulated analyst review",
        })

    append_reviews(review_path, simulated_rows)

    df = pd.read_csv(review_path)

    summary = (
        df.groupby("reviewer_action")
          .size()
          .to_frame("count")
    )

    report_path = paths.reports_dir / "review_simulation_preview.md"
    with open(report_path, "w") as f:
        f.write("# Review Simulation Preview\n\n")
        f.write("Simulated analyst feedback counts:\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")
        f.write("## Notes\n")
        f.write("- This simulates human-in-the-loop feedback.\n")
        f.write("- In production, these labels would arrive asynchronously.\n")

    print(f"Wrote {review_path}")
    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()