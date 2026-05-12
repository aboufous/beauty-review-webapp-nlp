"""Task 2 + Task 3 — product detail page with review form and similar items."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.data import append_review, product_row, reviews_for_product  # noqa: E402
from src.models.loader import load_predictor  # noqa: E402
from src.similarity import similar_products  # noqa: E402
from src.ui import cached_products, cached_reviews, placeholder_image, rating_stars, render_product_card  # noqa: E402

st.set_page_config(page_title="Product — GlowMate", page_icon="📦", layout="wide")

products = cached_products()
pid_raw = st.query_params.get("product_id")

if pid_raw is None:
    st.info(
        "Pick a product to view its detail, similar items, and write a review. "
        "You can also reach this page by clicking **View details** on the Browse page."
    )
    options = products.assign(
        _label=lambda d: d["brand_name"].fillna("(no brand)") + " — " + d["product_title"].fillna("(no title)")
    )
    pick = st.selectbox(
        "Choose a product",
        options=options["product_id"].tolist(),
        format_func=lambda pid: options.loc[options["product_id"] == pid, "_label"].iloc[0],
        index=None,
        placeholder="Type to filter by brand or product name…",
    )
    if pick is None:
        st.page_link("pages/1_🛍_Browse_and_Search.py", label="…or go to Browse", icon="🛍")
        st.stop()
    st.query_params["product_id"] = str(pick)
    st.rerun()

product = product_row(pid_raw, products)
if product is None:
    st.error(f"Product `{pid_raw}` not found.")
    st.query_params.clear()
    st.stop()

# === Header
top_l, top_r = st.columns([1, 2])
with top_l:
    st.image(placeholder_image(product["product_id"]), use_column_width=True)
with top_r:
    st.markdown(f"### {product.get('product_title') or '(untitled)'}")
    st.caption(f"**Brand:** {product.get('brand_name')}")
    cols = st.columns(3)
    cols[0].metric("Avg rating", rating_stars(product.get("avg_product_rating")))
    price = product.get("price")
    cols[1].metric("Price", f"₹{int(price):,}" if pd.notna(price) else "—")
    cols[2].metric("Reviews on file", f"{int(product.get('n_reviews') or 0):,}")
    if product.get("product_tags"):
        st.caption(f"Tags: {product['product_tags']}")
    if product.get("product_url"):
        st.markdown(f"[Original product page ↗]({product['product_url']})")

st.divider()

# === Task 2 — Write a review with fused-model label
st.subheader("📝 Write a review")
st.caption(
    "Submit a review and we'll predict whether it reads like it came from "
    "a verified **buyer** — the per-source breakdown below explains why."
)

with st.form("review_form", clear_on_submit=False):
    title = st.text_input("Review title", placeholder="Summarise your impression")
    body = st.text_area("Review text", height=150, placeholder="What did you actually think?")
    rating = st.slider("Your rating", 1, 5, 4)
    author = st.text_input("Your display name (optional)", placeholder="Anonymous")
    submitted = st.form_submit_button("Predict & continue", type="primary", use_container_width=True)

if submitted:
    if not body.strip():
        st.error("Review text is required.")
    else:
        predictor = load_predictor()
        pred = predictor.predict(
            review={"review_title": title, "review_text": body, "review_rating": rating},
            product=product,
        )
        st.session_state["pending_review"] = {
            "title": title,
            "body": body,
            "rating": rating,
            "author": author or "Anonymous",
            "pred": pred,
        }

pending = st.session_state.get("pending_review")
if pending:
    pred = pending["pred"]
    label_str = "✅ Likely **buyer**" if pred.label else "🚫 Likely **non-buyer**"
    st.markdown(f"#### Model prediction: {label_str}  ·  confidence {pred.proba:.0%}")
    st.progress(pred.proba)

    breakdown = pd.DataFrame(
        {
            "source": ["Review text", "Rating + metadata", "Product history prior"],
            "probability": [pred.proba_text, pred.proba_meta, pred.proba_prior],
            "weight in fusion": [pred.weight_text, pred.weight_meta, pred.weight_prior],
        }
    )
    st.dataframe(breakdown, hide_index=True, use_container_width=True)
    st.caption(
        "**HD-level fused architecture.** Three independently-trained models — "
        "TF-IDF(1,2) + LR on text · HistGradientBoosting on rating/price/brand · "
        "Bayesian-smoothed per-product prior — combined by a logistic-regression "
        "stacker fit on out-of-fold probabilities."
    )

    override = st.radio(
        "Final label to save",
        options=[
            f"Keep predicted label ({'buyer' if pred.label else 'non-buyer'})",
            f"Override to ({'non-buyer' if pred.label else 'buyer'})",
        ],
        horizontal=True,
    )
    user_overrode = override.startswith("Override")
    final_label = (not pred.label) if user_overrode else pred.label

    if st.button("Save review", type="primary"):
        rid = append_review({
            "product_id": product["product_id"],
            "brand_name": product.get("brand_name"),
            "review_title": pending["title"],
            "review_text": pending["body"],
            "author": pending["author"],
            "review_rating": pending["rating"],
            "is_a_buyer": bool(final_label),
            "product_title": product.get("product_title"),
            "price": product.get("price"),
            "avg_product_rating": product.get("avg_product_rating"),
            "product_rating_count": product.get("product_rating_count"),
            "product_tags": product.get("product_tags"),
            "product_url": product.get("product_url"),
            "predicted_label": bool(pred.label),
            "predicted_proba": float(pred.proba),
            "user_overrode": bool(user_overrode),
        })
        cached_reviews.clear()
        del st.session_state["pending_review"]
        st.query_params["product_id"] = str(product["product_id"])
        st.query_params["review_id"] = rid
        st.success(f"Saved! Review id **{rid}** — bookmark this URL to revisit.")
        st.rerun()

# === Existing reviews
st.divider()
st.subheader("💬 Reviews")
focus_rid = st.query_params.get("review_id")
review_df = reviews_for_product(product["product_id"], cached_reviews())
if focus_rid:
    pinned = review_df[review_df["review_id"].astype(str) == str(focus_rid)]
    if len(pinned):
        rest = review_df[review_df["review_id"].astype(str) != str(focus_rid)]
        review_df = pd.concat([pinned, rest], ignore_index=True)
        st.info(f"Showing your saved review (id **{focus_rid}**) at the top.")

show_n = min(len(review_df), 10)
st.caption(f"Showing {show_n} of {len(review_df):,} reviews.")
for _, r in review_df.head(show_n).iterrows():
    with st.container(border=True):
        title_str = r.get("review_title") or "(no title)"
        is_pinned = focus_rid and str(r.get("review_id")) == str(focus_rid)
        if is_pinned:
            st.markdown(f"**🆕 {title_str}**")
        else:
            st.markdown(f"**{title_str}**")
        st.caption(
            f"{r.get('author') or 'Anonymous'} · "
            f"rating {rating_stars(r.get('review_rating'))} · "
            f"{'buyer ✅' if bool(r.get('is_a_buyer')) else 'non-buyer 🚫'}"
        )
        st.write(r.get("review_text") or "—")
        if pd.notna(r.get("predicted_label")):
            label = "buyer" if bool(r["predicted_label"]) else "non-buyer"
            override = " (user overrode)" if bool(r.get("user_overrode")) else ""
            st.caption(f"_Model predicted: {label} @ {float(r.get('predicted_proba') or 0):.0%}{override}_")

# === Task 3 — Similar items
st.divider()
st.subheader("✨ Similar items you might like")
sim = similar_products(product["product_id"], products, k=6)
if len(sim) == 0:
    st.info("No similar items available.")
else:
    cols = st.columns(3)
    for i, (_, row) in enumerate(sim.iterrows()):
        with cols[i % 3]:
            render_product_card(row.to_dict(), score=row.get("_score"), key_prefix="sim")
