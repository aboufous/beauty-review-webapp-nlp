"""Task 4 — Admin analytics dashboard.

This page is intentionally shaped as an operations console, not just a chart
gallery. A real beauty retailer would use it to spot risky reviews, products
that need moderation, and model drift signals after shoppers submit feedback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.models.loader import METRICS_PATH  # noqa: E402
from src.ui import cached_reviews  # noqa: E402

st.set_page_config(page_title="Admin — GlowMate", page_icon="📊", layout="wide")

ACCENT_BLUE = "#2563eb"
ACCENT_TEAL = "#0f766e"
ACCENT_AMBER = "#d97706"
ACCENT_RED = "#dc2626"
ACCENT_GREEN = "#16a34a"


def _bool_rate(s: pd.Series) -> float:
    return s.astype("boolean").astype("float").mean()


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _prepare_reviews(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_date"] = pd.to_datetime(out["review_date"], errors="coerce", dayfirst=True)
    out["_created_at"] = pd.to_datetime(out.get("created_at"), errors="coerce", utc=True)
    out["_source"] = out["predicted_label"].notna().map(
        {True: "User submitted", False: "Seed dataset"}
    )
    out["_is_buyer"] = out["is_a_buyer"].astype("boolean")
    out["_is_non_buyer"] = out["_is_buyer"].eq(False)
    out["_rating_num"] = pd.to_numeric(out["review_rating"], errors="coerce")
    out["_predicted_bool"] = out["predicted_label"].astype("boolean")
    out["_model_confidence"] = pd.to_numeric(out["predicted_proba"], errors="coerce")
    out["_override"] = out["user_overrode"].astype("boolean").fillna(False)
    out["_review_len"] = out["review_text"].fillna("").astype(str).str.split().str.len()
    out["_snippet"] = (
        out["review_text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 180)
    )

    model_non_buyer = out["_predicted_bool"].eq(False) & out["_model_confidence"].ge(0.70)
    low_rating = out["_rating_num"].le(2)
    long_negative = out["_is_non_buyer"] & out["_review_len"].ge(35)
    out["_risk_score"] = (
        45 * out["_is_non_buyer"].fillna(False).astype(int)
        + 20 * low_rating.fillna(False).astype(int)
        + 20 * model_non_buyer.fillna(False).astype(int)
        + 10 * out["_override"].astype(int)
        + 5 * long_negative.fillna(False).astype(int)
    ).clip(0, 100)
    out["_priority"] = pd.cut(
        out["_risk_score"],
        bins=[-1, 39, 69, 100],
        labels=["Monitor", "Review soon", "Urgent"],
    ).astype(str)
    out["_suggested_action"] = "Monitor trend"
    out.loc[model_non_buyer, "_suggested_action"] = "Check buyer authenticity"
    out.loc[low_rating & out["_is_non_buyer"], "_suggested_action"] = "Customer recovery"
    out.loc[out["_override"], "_suggested_action"] = "Audit model disagreement"
    return out


def _metric_delta(current: float, baseline: float | None, fmt: str = "{:+.1f}") -> str | None:
    if baseline is None or pd.isna(baseline) or pd.isna(current):
        return None
    return fmt.format(current - baseline)


reviews = _prepare_reviews(cached_reviews())

st.title("📊 Admin analytics dashboard")
st.caption(
    "Operational view over reviews, product risk, and model feedback. "
    "Use it to prioritise moderation and customer recovery, not just to admire charts."
)

st.sidebar.header("Filters")
date_min = reviews["_date"].min()
date_max = reviews["_date"].max()
if pd.notna(date_min) and pd.notna(date_max):
    date_range = st.sidebar.date_input(
        "Review date range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )
else:
    date_range = ()

brand_options = sorted(reviews["brand_name"].dropna().astype(str).unique().tolist())
selected_brands = st.sidebar.multiselect("Brands", brand_options, default=brand_options)
source_options = ["Seed dataset", "User submitted"]
selected_sources = st.sidebar.multiselect("Review source", source_options, default=source_options)
buyer_filter = st.sidebar.radio(
    "Final label",
    ["All", "Verified buyer", "Non-buyer / unverified"],
    horizontal=True,
)
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 0, step=5)

filtered = reviews.copy()
if len(date_range) == 2:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered["_date"].isna()) | ((filtered["_date"] >= start) & (filtered["_date"] < end))]
if selected_brands:
    filtered = filtered[filtered["brand_name"].astype(str).isin(selected_brands)]
if selected_sources:
    filtered = filtered[filtered["_source"].isin(selected_sources)]
if buyer_filter == "Verified buyer":
    filtered = filtered[filtered["_is_buyer"].eq(True)]
elif buyer_filter == "Non-buyer / unverified":
    filtered = filtered[filtered["_is_buyer"].eq(False)]
filtered = filtered[filtered["_risk_score"].ge(min_risk)]

baseline_buyer_rate = _bool_rate(reviews["is_a_buyer"])
buyer_rate = _bool_rate(filtered["is_a_buyer"]) if len(filtered) else float("nan")
avg_rating = filtered["_rating_num"].mean() if len(filtered) else float("nan")
user_rows_all = reviews[reviews["predicted_label"].notna()]
user_rows = filtered[filtered["predicted_label"].notna()]
override_rate = _bool_rate(user_rows["user_overrode"]) if len(user_rows) else float("nan")
urgent_count = int(filtered["_priority"].eq("Urgent").sum())
non_buyer_rate = 1 - buyer_rate if pd.notna(buyer_rate) else float("nan")

st.subheader("Store health")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews in view", f"{len(filtered):,}", delta=f"{len(filtered) - len(reviews):+,}")
k2.metric(
    "Verified buyer rate",
    f"{buyer_rate:.0%}" if pd.notna(buyer_rate) else "—",
    delta=_metric_delta(buyer_rate * 100, baseline_buyer_rate * 100, "{:+.1f} pp")
    if pd.notna(buyer_rate)
    else None,
)
k3.metric("Avg rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "—")
k4.metric(
    "Model override rate",
    f"{override_rate:.0%}" if pd.notna(override_rate) else "—",
    help="Share of user-submitted reviews where the user changed the model label.",
)
k5.metric(
    "Urgent review queue",
    f"{urgent_count:,}",
    delta=f"{non_buyer_rate:.0%} non-buyer / unverified" if pd.notna(non_buyer_rate) else None,
)

st.download_button(
    "Download filtered review data",
    data=_csv_bytes(filtered.drop(columns=[c for c in filtered.columns if c.startswith("_")])),
    file_name="glowmate_filtered_reviews.csv",
    mime="text/csv",
)

st.divider()

st.subheader("Model quality snapshot")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    st.metric("Active fused model — Macro-F1", f"{metrics['test_macro_f1']['fused']:.3f}")
    st.caption(
        "The active app prediction is the DI/HD fusion model. "
        "Task 3 notebook best single model Macro-F1 = 0.7110 and is retained as benchmark documentation."
    )
    cols = st.columns(3)
    for col, source in zip(cols, ["text", "meta", "fused"]):
        col.metric(
            f"{source.title()} model — Macro-F1",
            f"{metrics['test_macro_f1'][source]:.3f}",
            delta=f"acc {metrics['test_accuracy'][source]:.3f}",
        )
    w = metrics["fusion_weights"]
    st.caption(
        f"Fusion weights — text: **{w['text']:.0%}** · meta: **{w['meta']:.0%}**"
    )
else:
    st.warning("`models/metrics.json` missing — run `python scripts/train.py`.")

st.divider()

tabs = st.tabs(["Operations queue", "Brand & product watchlist", "Trends", "Model feedback"])

with tabs[0]:
    st.markdown("**Review moderation queue**")
    queue = (
        filtered.sort_values(["_risk_score", "_created_at", "_date"], ascending=[False, False, False])
        .head(50)
        .loc[
            :,
            [
                "_priority",
                "_risk_score",
                "_suggested_action",
                "brand_name",
                "product_title",
                "review_rating",
                "is_a_buyer",
                "_source",
                "_snippet",
                "review_id",
            ],
        ]
        .rename(
            columns={
                "_priority": "priority",
                "_risk_score": "risk_score",
                "_suggested_action": "suggested_action",
                "_source": "source",
                "_snippet": "review_snippet",
            }
        )
    )
    if len(queue):
        st.dataframe(
            queue,
            hide_index=True,
            use_container_width=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "risk_score", min_value=0, max_value=100, format="%d"
                ),
            },
        )
        st.download_button(
            "Download moderation queue",
            data=_csv_bytes(queue),
            file_name="glowmate_moderation_queue.csv",
            mime="text/csv",
        )
    else:
        st.info("No reviews match the current queue filters.")

    st.markdown("**Newest user submissions**")
    recent_submissions = (
        user_rows.sort_values("_created_at", ascending=False)
        .head(10)
        .loc[
            :,
            [
                "_created_at",
                "brand_name",
                "product_title",
                "review_rating",
                "predicted_label",
                "predicted_proba",
                "is_a_buyer",
                "user_overrode",
                "_snippet",
            ],
        ]
        .rename(columns={"_created_at": "submitted_at", "_snippet": "review_snippet"})
    )
    if len(recent_submissions):
        st.dataframe(recent_submissions, hide_index=True, use_container_width=True)
    else:
        st.info("No user-submitted reviews in the current filter.")

with tabs[1]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Brand risk matrix**")
        brand_risk = (
            filtered.groupby("brand_name", dropna=False)
            .agg(
                reviews=("review_id", "count"),
                avg_rating=("_rating_num", "mean"),
                buyer_rate=("is_a_buyer", _bool_rate),
                avg_risk=("_risk_score", "mean"),
                urgent=("_priority", lambda s: int((s == "Urgent").sum())),
            )
            .query("reviews >= 20")
            .reset_index()
        )
        if len(brand_risk):
            fig = px.scatter(
                brand_risk,
                x="buyer_rate",
                y="avg_rating",
                size="reviews",
                color="avg_risk",
                hover_name="brand_name",
                hover_data={"reviews": True, "urgent": True, "buyer_rate": ":.0%"},
                color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
                labels={
                    "buyer_rate": "Verified buyer rate",
                    "avg_rating": "Average rating",
                    "avg_risk": "Risk score",
                },
            )
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough reviews per brand after filtering.")

    with c2:
        st.markdown("**Product watchlist**")
        product_risk = (
            filtered.groupby(["product_id", "brand_name", "product_title"], dropna=False)
            .agg(
                reviews=("review_id", "count"),
                buyer_rate=("is_a_buyer", _bool_rate),
                avg_rating=("_rating_num", "mean"),
                avg_risk=("_risk_score", "mean"),
                urgent=("_priority", lambda s: int((s == "Urgent").sum())),
            )
            .reset_index()
        )
        product_risk = product_risk[product_risk["reviews"].ge(10)]
        product_risk = product_risk.sort_values(
            ["avg_risk", "urgent", "reviews"], ascending=[False, False, False]
        ).head(20)
        if len(product_risk):
            display = product_risk.assign(
                product_link=lambda d: d["product_id"].map(
                    lambda pid: f"/product_detail?product_id={pid}"
                )
            )[
                [
                    "brand_name",
                    "product_title",
                    "reviews",
                    "buyer_rate",
                    "avg_rating",
                    "avg_risk",
                    "urgent",
                ]
            ]
            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "buyer_rate": st.column_config.NumberColumn("buyer_rate", format="%.0%%"),
                    "avg_rating": st.column_config.NumberColumn("avg_rating", format="%.2f"),
                    "avg_risk": st.column_config.ProgressColumn(
                        "avg_risk", min_value=0, max_value=100, format="%.0f"
                    ),
                },
            )
        else:
            st.info("No product has enough filtered reviews for a stable watchlist.")

    if len(brand_risk):
        st.markdown("**Recommended admin actions**")
        weakest_brand = brand_risk.sort_values(["avg_risk", "buyer_rate"], ascending=[False, True]).iloc[0]
        strongest_brand = brand_risk.sort_values(["buyer_rate", "avg_rating"], ascending=[False, False]).iloc[0]
        a1, a2, a3 = st.columns(3)
        a1.info(
            f"Prioritise **{weakest_brand['brand_name']}**: "
            f"{weakest_brand['urgent']} urgent reviews and {weakest_brand['buyer_rate']:.0%} verified buyer rate."
        )
        a2.success(
            f"Use **{strongest_brand['brand_name']}** as a healthy benchmark: "
            f"{strongest_brand['buyer_rate']:.0%} verified buyer rate, {strongest_brand['avg_rating']:.2f} avg rating."
        )
        a3.warning(
            "Audit products with high risk but high review volume first; they affect the most shoppers."
        )

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rating distribution**")
        fig = px.histogram(
            filtered,
            x="_rating_num",
            color="_is_buyer",
            barmode="group",
            nbins=5,
            labels={"_is_buyer": "Verified buyer?", "_rating_num": "Rating"},
            color_discrete_map={True: ACCENT_BLUE, False: ACCENT_AMBER},
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top 15 brands by review volume**")
        top_brands = (
            filtered.groupby("brand_name", dropna=False)
            .agg(n=("review_id", "count"), buyer_rate=("is_a_buyer", _bool_rate))
            .nlargest(15, "n")
            .reset_index()
        )
        if len(top_brands):
            fig = px.bar(
                top_brands,
                x="n",
                y="brand_name",
                orientation="h",
                color="buyer_rate",
                color_continuous_scale=["#f59e0b", "#14b8a6", "#2563eb"],
                labels={"n": "Reviews", "brand_name": "Brand", "buyer_rate": "Verified buyer rate"},
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No brands match the current filters.")

    st.markdown("**Review volume + buyer-rate over time**")
    ts = (
        filtered.dropna(subset=["_date"])
        .assign(month=lambda d: d["_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(volume=("review_id", "count"), buyer_rate=("is_a_buyer", _bool_rate))
        .reset_index()
    )
    if len(ts):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=ts["month"],
                y=ts["volume"],
                name="Reviews per month",
                mode="lines",
                fill="tozeroy",
                line=dict(color=ACCENT_BLUE),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=ts["month"],
                y=ts["buyer_rate"],
                name="Verified buyer rate",
                mode="lines",
                line=dict(color=ACCENT_AMBER, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Reviews per month", secondary_y=False)
        fig.update_yaxes(title_text="Verified buyer rate", range=[0, 1], tickformat=".0%", secondary_y=True)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No parseable review dates in the current filters.")

with tabs[3]:
    st.markdown("**Predicted vs. final label — user-submitted reviews only**")
    if len(user_rows):
        conf = (
            user_rows.assign(
                predicted=user_rows["_predicted_bool"].map({True: "buyer", False: "non-buyer"}),
                final=user_rows["_is_buyer"].map({True: "buyer", False: "non-buyer"}),
            )
            .groupby(["predicted", "final"])
            .size()
            .reset_index(name="count")
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.density_heatmap(
                conf,
                x="predicted",
                y="final",
                z="count",
                text_auto=True,
                color_continuous_scale=["#e0f2fe", "#38bdf8", "#0369a1"],
            )
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            confidence_bins = user_rows.assign(
                confidence_band=pd.cut(
                    user_rows["_model_confidence"],
                    bins=[0, 0.6, 0.8, 1.0],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True,
                )
            )
            band_summary = (
                confidence_bins.groupby("confidence_band", observed=False)
                .agg(
                    submissions=("review_id", "count"),
                    override_rate=("user_overrode", _bool_rate),
                    avg_confidence=("_model_confidence", "mean"),
                )
                .reset_index()
            )
            fig = px.bar(
                band_summary,
                x="confidence_band",
                y="override_rate",
                text="submissions",
                color="avg_confidence",
                color_continuous_scale=["#fef3c7", "#f59e0b", "#b45309"],
                labels={
                    "confidence_band": "Model confidence band",
                    "override_rate": "Override rate",
                    "avg_confidence": "Avg confidence",
                },
            )
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Disagreement audit list**")
        disagreements = (
            user_rows[user_rows["_override"]]
            .sort_values("_model_confidence", ascending=False)
            .head(25)
            .loc[
                :,
                [
                    "_model_confidence",
                    "brand_name",
                    "product_title",
                    "review_rating",
                    "predicted_label",
                    "is_a_buyer",
                    "_snippet",
                    "review_id",
                ],
            ]
            .rename(columns={"_model_confidence": "model_confidence", "_snippet": "review_snippet"})
        )
        if len(disagreements):
            st.dataframe(
                disagreements,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "model_confidence": st.column_config.NumberColumn(
                        "model_confidence", format="%.0%%"
                    )
                },
            )
        else:
            st.success("No model disagreements in the current filter.")
    else:
        st.info("No user-submitted reviews yet — write one on a product page to populate this panel.")
