import sys
from pathlib import Path
from src.app.io import load_triage_queue, load_reviews, reviews_for_transaction

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st

from src.config import Paths
from src.app.io import load_triage_queue, load_reviews
from src.app.review_actions import mark_reviewed, reviewed_set
st.set_page_config(page_title="Fraud Triage Review Queue", layout="wide")

paths = Paths()
queue_path = paths.data_dir / "triage_queue_test.csv"
review_path = paths.data_dir / "reviews.csv"

st.title("Fraud Detection — Human Review Queue")

# Load data
queue = load_triage_queue(queue_path)
# Normalize columns to prevent silent filter mismatches
queue["decision_band"] = queue["decision_band"].astype(str).str.strip().str.lower()
queue["risk_score"] = queue["risk_score"].astype(float)
queue["transaction_index"] = queue["transaction_index"].astype(int)
reviews = load_reviews(review_path)
st.sidebar.header("Debug")
st.sidebar.write("Rows in queue:", len(queue))
st.sidebar.write("Decision band value counts (raw):")
st.sidebar.write(queue["decision_band"].value_counts(dropna=False))

st.sidebar.write("Risk score dtype:", queue["risk_score"].dtype)
st.sidebar.write("Risk score min/max:", float(queue["risk_score"].min()), float(queue["risk_score"].max()))
already_reviewed = reviewed_set(reviews)

# Filters
st.sidebar.header("Filters")
band = st.sidebar.multiselect(
    "Decision band",
    options=["review", "decline", "approve"],
    default=["review", "decline"]
)

min_score = st.sidebar.slider("Min risk score", 0.0, 1.0, 0.0, 0.01)
only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)

df = queue.copy()
df = df[df["decision_band"].isin(band)]
df = df[df["risk_score"] >= min_score]
st.sidebar.write("Rows after filters:", len(df))
if only_unreviewed:
    df = df[~df["transaction_index"].astype(int).isin(already_reviewed)]

df = df.sort_values(["decision_band", "risk_score"], ascending=[True, False])

st.subheader("Queue")
cols = ["transaction_index", "risk_score", "decision_band", "reason_1", "reason_2", "reason_3"]
st.dataframe(df[cols].head(300), use_container_width=True)

st.divider()

st.subheader("Review a transaction")
tx_id = st.selectbox(
    "Select transaction_index",
    options=df["transaction_index"].astype(int).head(300).tolist() if len(df) else []
)

if tx_id is not None and len(df):
    row = queue[queue["transaction_index"].astype(int) == int(tx_id)].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", float(row["risk_score"]))
    c2.metric("Decision band", str(row["decision_band"]))
    c3.metric("Reviewed?", "Yes" if int(tx_id) in already_reviewed else "No")
history = reviews_for_transaction(reviews, tx_id)

    if not history.empty:
        st.subheader("Review history")
        st.dataframe(
            history[[
                "review_timestamp_utc",
                "reviewer_action",
                "reviewer_notes"
            ]].sort_values("review_timestamp_utc", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No prior reviews for this transaction.")

    st.write("**Top reasons:**")
    st.write([row.get("reason_1"), row.get("reason_2"), row.get("reason_3")])

    with st.form("review_form"):
        action = st.selectbox("Reviewer action", ["confirmed_fraud", "false_positive", "needs_more_info"])
        notes = st.text_area("Notes", placeholder="What did you observe? Why this decision?")
        submitted = st.form_submit_button("Submit review")

        if submitted:
            mark_reviewed(
                review_path=review_path,
                transaction_index=int(tx_id),
                risk_score=float(row["risk_score"]),
                decision_band=str(row["decision_band"]),
                reviewer_action=action,
                reviewer_notes=notes
            )
            st.success("Saved review to reviews.csv")
            st.rerun()

st.divider()
st.subheader("Review stats")
st.write(reviews["reviewer_action"].value_counts(dropna=False))