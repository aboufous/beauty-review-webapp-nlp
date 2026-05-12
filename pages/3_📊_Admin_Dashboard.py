"""Task 4 — Admin analytics dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.loader import METRICS_PATH  # noqa: E402
from src.ui import cached_reviews  # noqa: E402

st.set_page_config(page_title="Admin — GlowMate", page_icon="📊", layout="wide")

st.title("📊 Admin analytics dashboard")
st.caption(
    "Live view over `data/reviews.csv` — refreshes whenever a new review is saved. "
    "The **Predicted-vs-final** chart is the unique-to-this-app signal: it measures "
    "how often users override the model."
)

reviews = cached_reviews()
reviews["is_a_buyer"] = reviews["is_a_buyer"].astype("boolean")

# === Model metrics card
st.subheader("Held-out test metrics")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    cols = st.columns(4)
    for col, source in zip(cols, ["text", "meta", "prior", "fused"]):
        col.metric(
            f"{source.title()} model — Macro-F1",
            f"{metrics['test_macro_f1'][source]:.3f}",
            delta=f"acc {metrics['test_accuracy'][source]:.3f}",
        )
    w = metrics["fusion_weights"]
    st.caption(
        f"Fusion weights — text: **{w['text']:.0%}** · meta: **{w['meta']:.0%}** · prior: **{w['prior']:.0%}**"
    )
else:
    st.warning("`models/metrics.json` missing — run `python scripts/train.py`.")

st.divider()

# === Charts row 1
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Rating distribution**")
    fig = px.histogram(
        reviews,
        x="review_rating",
        color="is_a_buyer",
        barmode="group",
        nbins=5,
        labels={"is_a_buyer": "Buyer?", "review_rating": "Rating"},
        color_discrete_map={True: "#3b82f6", False: "#f97316"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("**Top 15 brands by review volume**")
    top_brands = (
        reviews.groupby("brand_name", dropna=False)
        .agg(n=("review_id", "count"), buyer_rate=("is_a_buyer", lambda s: s.astype("float").mean()))
        .nlargest(15, "n")
        .reset_index()
    )
    fig = px.bar(
        top_brands,
        x="n",
        y="brand_name",
        orientation="h",
        color="buyer_rate",
        color_continuous_scale="Blues",
        labels={"n": "Reviews", "brand_name": "Brand", "buyer_rate": "Buyer rate"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# === Charts row 2
st.markdown("**Review volume + buyer-rate over time**")
reviews["_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")
ts = (
    reviews.dropna(subset=["_date"])
    .assign(month=lambda d: d["_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")
    .agg(volume=("review_id", "count"), buyer_rate=("is_a_buyer", lambda s: s.astype("float").mean()))
    .reset_index()
)
if len(ts):
    fig = px.area(ts, x="month", y="volume", labels={"volume": "Reviews per month"})
    fig.add_scatter(x=ts["month"], y=ts["buyer_rate"] * ts["volume"].max(), name="Buyer rate (scaled)", mode="lines")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No parseable `review_date` values.")

# === Predicted vs final
st.markdown("**Predicted vs. final label** — user-submitted reviews only")
user_rows = reviews.dropna(subset=["predicted_label"])
if len(user_rows):
    conf = (
        user_rows.assign(
            predicted=user_rows["predicted_label"].astype("boolean").map({True: "buyer", False: "non-buyer"}),
            final=user_rows["is_a_buyer"].astype("boolean").map({True: "buyer", False: "non-buyer"}),
        )
        .groupby(["predicted", "final"])
        .size()
        .reset_index(name="count")
    )
    fig = px.density_heatmap(
        conf, x="predicted", y="final", z="count", text_auto=True, color_continuous_scale="Purples"
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    override_rate = user_rows["user_overrode"].astype("boolean").astype("float").mean()
    st.metric("User override rate", f"{override_rate:.0%}", help="Share of submissions where the user disagreed with the model")
else:
    st.info("No user-submitted reviews yet — write one on a product page to populate this chart.")
