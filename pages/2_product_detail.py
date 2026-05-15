"""Task 2 + Task 3 — product detail page with review form and similar items."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src.data import (append_review, product_row, reviews_for_product)
from src.models.loader import load_predictor, load_sentiment_model  # noqa: E402
from src.recommender import has_dense_vectors  # noqa: E402
from src.similarity import similar_products  # noqa: E402
from src.ui import (cached_products, cached_reviews,  # noqa: E402
					product_image, rating_stars, render_product_card)

st.set_page_config(page_title="Product — GlowMate", page_icon="📦", layout="wide")


def review_signature(title: str, body: str, rating: int) -> tuple[str, str, int]:
	"""Stable identity for the review currently in the form."""
	return title.strip(), body.strip(), int(rating)


def scroll_to_anchor(anchor_id: str):
	components.html(
		f"""
        <script>
        const anchor = window.parent.document.getElementById("{anchor_id}");
        if (anchor) {{
            anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
        """,
		height=0,
	)


def render_prediction_panel(pending: dict, product: dict):
	pred = pending["pred"]
	sentiment_model = load_sentiment_model()
	sentiment, sentiment_confidence, action = sentiment_model.label_one(
		f"{pending['title']} {pending['body']}",
		pending["rating"],
	)
	st.markdown('<div id="prediction-result-anchor"></div>', unsafe_allow_html=True)
	if st.session_state.pop("scroll_to_prediction", False):
		scroll_to_anchor("prediction-result-anchor")

	with st.container(border=True):
		st.markdown("#### Prediction result")
		st.caption("This review has not been saved yet. Confirm the final label below.")
		label_str = "✅ Likely **verified buyer**" if pred.label else "🚫 Likely **non-buyer / unverified**"
		st.markdown(f"**Verified-buyer prediction:** {label_str}  ·  confidence **{pred.proba:.0%}**")
		st.progress(pred.proba)

		s1, s2 = st.columns(2)
		with s1:
			st.caption("CUSTOMER SENTIMENT")
			st.markdown(f"### {sentiment.lower()}")
			st.caption(f"{sentiment_confidence:.0%} CONFIDENCE")
		with s2:
			st.caption("SUGGESTED ACTION")
			st.markdown(f"### {action.lower()}")
		if pred.label and sentiment == "Negative":
			st.warning(
				"This is a negative review from someone who still appears to be a real buyer. "
				"Save it as a verified buyer review, then prioritise customer recovery."
			)

		breakdown = pd.DataFrame(
			{
				"source": ["Review text", "Rating + metadata", "Product history prior"],
				"probability": [pred.proba_text, pred.proba_meta, pred.proba_prior],
				"relative weight": [pred.weight_text, pred.weight_meta, pred.weight_prior],
				"data type": [
					"TF-IDF text from review title/body",
					"Tabular review + product metadata",
					"Bayesian-smoothed product/brand behaviour",
				],
			}
		)
		st.dataframe(breakdown, hide_index=True, use_container_width=True)
		st.caption(
			"**HD final fusion model.** The final verified-buyer prediction fuses "
			"three independently trained models that use different data types: text, "
			"structured metadata, and product/brand history. This matches the DI/HD "
			"requirement for a fused final result."
		)

		override = st.radio(
			"Final label to save",
			options=[
				f"Keep predicted label ({'verified buyer' if pred.label else 'non-buyer / unverified'})",
				f"Override to ({'non-buyer / unverified' if pred.label else 'verified buyer'})",
			],
			horizontal=True,
			key="final_label_to_save",
		)
		user_overrode = override.startswith("Override")
		final_label = (not pred.label) if user_overrode else pred.label

		if st.button("Save review", type="primary", key="save_review_prediction"):
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


products = cached_products()
if st.query_params.get("clear_product") == "1":
	st.session_state.pop("current_product_id", None)
	st.session_state.pop("pending_review", None)
	st.query_params.clear()

pid_raw = st.query_params.get("product_id")
if pid_raw is None and "selected_product_id" in st.session_state:
	pid_raw = st.session_state.pop("selected_product_id")
	st.query_params["product_id"] = str(pid_raw)
if pid_raw is None and "current_product_id" in st.session_state:
	pid_raw = st.session_state["current_product_id"]
	st.query_params["product_id"] = str(pid_raw)
if (
	pid_raw is None
	and st.session_state.get("scroll_to_prediction", False)
	and "pending_review" in st.session_state
):
	pid_raw = st.session_state["pending_review"].get("product_id")
	if pid_raw is not None:
		st.query_params["product_id"] = str(pid_raw)

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
		st.link_button("🛍 …or go to Browse", "/browse_and_search")
		st.stop()
	st.query_params["product_id"] = str(pick)
	st.rerun()

product = product_row(pid_raw, products)
if product is None:
	st.error(f"Product {pid_raw} not found.")
	st.query_params.clear()
	st.stop()

st.query_params["product_id"] = str(product["product_id"])
st.session_state["current_product_id"] = str(product["product_id"])

top_l, top_r = st.columns([1, 2])
with top_l:
	st.image(
		product_image(product["product_id"], title=product.get("product_title")),
		use_column_width=True,
	)
with top_r:
	st.markdown(f"### {product.get('product_title') or '(untitled)'}")
	st.caption(f"**Brand: ** {product.get('brand_name')}")
	cols = st.columns(3)
	cols[0].metric("Avg rating", rating_stars(product.get("avg_product_rating")))
	price = product.get("price")
	cols[1].metric("Price", f"₹{int(price):,}" if pd.notna(price) else "—")
	cols[2].metric("Reviews on file", f"{int(product.get('n_reviews') or 0):,}")
	if product.get("product_tags"):
		st.caption(f"Tags: {product['product_tags']}")
	if product.get("product_url"):
		st.markdown(f"[Original product page ↗]({product['product_url']})")
	st.link_button("Choose another product", "/product_detail?clear_product=1")

st.divider()

st.subheader("📝 Write a review")
st.caption(
	"Submit a review and we'll predict whether it reads like it came from "
	"a **verified buyer / authentic purchaser**. Customer sentiment is shown "
	"separately, because a real buyer can still leave a negative review or say "
	"they would not buy again."
)

pending = st.session_state.get("pending_review")
if pending and pending.get("product_id") != str(product["product_id"]):
	del st.session_state["pending_review"]
	pending = None

with st.form("review_form", clear_on_submit=False):
	title = st.text_input(
		"Review title (optional)",
		placeholder="Summarise your impression",
		key="review_title_input",
	)
	body = st.text_area(
		"Review text",
		height=150,
		placeholder="What did you actually think? (required)",
		key="review_body_input",
	)
	rating = st.slider("Your rating", 1, 5, 4, key="review_rating_input")
	author = st.text_input(
		"Your display name (optional)",
		placeholder="Anonymous",
		key="review_author_input",
	)
	submitted = st.form_submit_button("Predict & continue", type="primary", use_container_width=True)

current_review_signature = review_signature(title, body, rating)
prediction_is_current = bool(
	pending and pending.get("signature") == current_review_signature
)

if pending and prediction_is_current:
	render_prediction_panel(pending, product)
elif pending:
	st.warning(
		"The review text or rating changed after the last prediction. "
		"Click **Predict & continue** again to refresh the result."
	)

if submitted:
	if not body.strip():
		st.error("Review text is required.")
	else:
		try:
			with st.spinner("Running the HD fusion buyer-authenticity model..."):
				predictor = load_predictor()
				pred = predictor.predict(
					review={"review_title": title, "review_text": body, "review_rating": rating},
					product=product,
				)
			st.session_state["pending_review"] = {
				"product_id": str(product["product_id"]),
				"title": title,
				"body": body,
				"rating": rating,
				"author": author or "Anonymous",
				"pred": pred,
				"signature": current_review_signature,
			}
			st.session_state["scroll_to_prediction"] = True
			st.query_params["product_id"] = str(product["product_id"])
			st.success("Prediction ready. Review the result below, then save or override it.")
			st.rerun()
		except Exception as exc:
			st.error(f"Prediction failed: {exc}")

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
			f"{'verified buyer ✅' if bool(r.get('is_a_buyer')) else 'non-buyer / unverified 🚫'}"
		)
		st.write(r.get("review_text") or "—")
		if pd.notna(r.get("predicted_label")):
			label = "verified buyer" if bool(r["predicted_label"]) else "non-buyer / unverified"
			override = " (user overrode)" if bool(r.get("user_overrode")) else ""
			st.caption(f"_Model predicted: {label} @ {float(r.get('predicted_proba') or 0):.0%}{override}_")

# === Task 3 — Similar items (HD-level hybrid recommender)
st.divider()
st.subheader("✨ Similar items you might like")
st.caption(
	"Hybrid recommender — fuses lexical (TF-IDF), semantic (FastText), "
	"attribute (brand & category) and numeric (price & rating) similarity, "
	"with MMR diversity re-ranking. See notebooks/Task3_Recommendation.ipynb."
)

if not has_dense_vectors():
	st.warning(
		"Semantic embeddings not built — running in 3-signal fallback "
		"(lexical + attribute + numeric). "
		"Run `python scripts/build_recommender.py` to enable the full hybrid."
	)

with st.expander("Recommender controls", expanded=False):
	sim_k = st.slider("How many recommendations", 3, 12, 6)
	sim_mmr = st.toggle(
		"Diversity re-ranking (MMR)",
		value=True,
		help="On: MMR penalises near-duplicates so the list shows variety.",
	)
	if sim_mmr:
		sim_lambda = st.slider(
			"λ  —  relevance vs diversity",
			min_value=0.0, max_value=1.0, value=0.7, step=0.05,
			help="1.0 = pure relevance · 0.0 = pure diversity · 0.7 = default.",
		)
		st.caption(
			f"At λ = **{sim_lambda:.2f}** the engine puts **{sim_lambda:.0%}** weight on "
			f"relevance and **{1 - sim_lambda:.0%}** on diversity. "
			"Drag toward 0.0 to force more variety, toward 1.0 to disable the diversity push."
		)
	else:
		sim_lambda = 0.7
		st.caption(
			"Toggle MMR on to expose the relevance ↔ diversity trade-off slider. "
			"With MMR off, the list collapses to close variants of the same product line."
		)

sim = similar_products(
	product["product_id"], products, k=sim_k,
	use_mmr=sim_mmr, mmr_lambda=sim_lambda,
)
if len(sim) == 0:
	st.info("No similar items available.")
else:
	cols = st.columns(3)
	for i, (_, row) in enumerate(sim.iterrows()):
		with cols[i % 3]:
			render_product_card(row.to_dict(), score=row.get("_score"), key_prefix="sim")
			with st.expander("Why is this similar?"):
				bc = st.columns(4)
				bc[0].metric("Semantic", f"{row.get('_score_semantic', 0):.2f}")
				bc[1].metric("Lexical", f"{row.get('_score_lexical', 0):.2f}")
				bc[2].metric("Attribute", f"{row.get('_score_attribute', 0):.2f}")
				bc[3].metric("Numeric", f"{row.get('_score_numeric', 0):.2f}")
				st.caption(
					"Final = 0.50·semantic + 0.20·lexical + 0.20·attribute + 0.10·numeric, "
					"then MMR re-ranked (λ=0.7) for diversity."
				)
