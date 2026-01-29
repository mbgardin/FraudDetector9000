from pathlib import Path
import numpy as np
import pandas as pd

from src.config import Paths
from src.review import append_reviews, init_review_store

def simulate_actions(df: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []

    for _, r in df.iterrows():
        # ground truth available here only for simulation
        is_fraud = int(r.get("is_fraud", 0))

        # realistic-ish reviewer behavior:
        # - if actually fraud: usually confirmed, sometimes "needs more info"
        # - if not fraud: usually false positive, sometimes "needs more info"
        if is_fraud == 1:
            action = rng.choice(["confirmed_fraud", "needs_more_info", "false_positive"], p=[0.85, 0.10, 0.05])
        else:
            action = rng.choice(["false_positive", "needs_more_info", "confirmed_fraud"], p=[0.88, 0.10, 0.02])

        note = "Simulated analyst review (prototype)."

        rows.append({
            "transaction_index": int(r["transaction_index"]),
            "risk_score": float(r["risk_score"]),
            "decision_band": str(r["decision_band"]),
            "reviewer_action": str(action),
            "reviewer_notes": note,
        })

    return rows

def main():
    paths = Paths()
    queue_path = paths.data_dir / "triage_queue_test.csv"
    review_path = paths.data_dir / "reviews.csv"
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    if not queue_path.exists():
        raise FileNotFoundError(
            f"Missing {queue_path}. Run: python -m scripts.build_triage_queue_with_reasons"
        )

    init_review_store(review_path)

    queue = pd.read_csv(queue_path)
    # If you saved index=True, pandas will likely create an "Unnamed: 0" col. Normalize it:
    if "transaction_index" not in queue.columns:
        if "Unnamed: 0" in queue.columns:
            queue = queue.rename(columns={"Unnamed: 0": "transaction_index"})
        else:
            raise ValueError("Could not find transaction_index column in triage queue.")

    # What humans actually see: mostly REVIEW band (declines are auto-blocked but could be audited)
    review_queue = queue[queue["decision_band"].isin(["review"])].copy()

    # Sample a manageable batch to simulate
    batch_size = min(200, len(review_queue))
    batch = review_queue.sample(n=batch_size, random_state=42)

    rows = simulate_actions(batch, seed=42)
    append_reviews(review_path, rows)

    reviews = pd.read_csv(review_path)
    summary = reviews["reviewer_action"].value_counts().to_frame("count")

    report_path = paths.reports_dir / "review_simulation_preview.md"
    with open(report_path, "w") as f:
        f.write("# Review Simulation Preview\n\n")
        f.write(f"- Loaded triage queue: `{queue_path.name}`\n")
        f.write(f"- Appended {len(rows)} simulated reviews to: `{review_path.name}`\n\n")
        f.write("## Reviewer action counts (all stored reviews)\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")
        f.write("## Notes\n")
        f.write("- This is a prototype simulation to prove the human-in-the-loop data contract.\n")
        f.write("- In production, reviewer actions arrive asynchronously and may disagree with ground truth.\n")

    print(f"Wrote {review_path}")
    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()