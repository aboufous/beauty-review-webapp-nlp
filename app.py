"""GlowMate Beauty Store — landing page."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="GlowMate Beauty Store", page_icon="💄", layout="wide")

st.title("💄 GlowMate Beauty Store")
st.caption(
    "NLP-powered cosmetics review and recommendation site — "
    "Milestone II for COSC3801/3015 Assignment 3 (MAI_Group 5)."
)

st.write(
    """
**What this app does**

- 🛍 **Browse & search** a catalogue of 295 cosmetics products with fuzzy
  keyword matching (tolerates typos and word-order changes).
- 📝 **Write a review** for any product — a DI/HD fusion model combines
  text, structured metadata, and product-history signals to predict whether
  the review reads like it came from a verified buyer.
- ✨ **Similar items** are surfaced via TF-IDF cosine similarity on every
  product page.
- 📊 An **admin dashboard** summarises sentiment, brand patterns, and how often
  shoppers override the model.
"""
)

st.subheader("Pick a place to start")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("pages/1_browse_and_search.py", label="**Browse the catalogue**", icon="🛍")
    st.caption("Search by brand, product name, or tag.")
with c2:
    st.page_link("pages/2_product_detail.py", label="**Open a product (after browsing)**", icon="📦")
    st.caption("Write a review and see the fused-model label.")
with c3:
    st.page_link("pages/3_admin_dashboard.py", label="**Admin dashboard**", icon="📊")
    st.caption("Sentiment, brand trends, override rate.")

# Surface the trained metrics on the landing page so reviewers can see them at a glance.
metrics_path = ROOT / "models" / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    with st.expander("Trained-model metrics"):
        st.metric("Active fused model — Macro-F1", f"{metrics['test_macro_f1']['fused']:.3f}")
        st.caption("Task 3 notebook best single model: Macro-F1 0.7110, kept as benchmark documentation.")
        cols = st.columns(3)
        for col, src in zip(cols, ["text", "meta", "fused"]):
            col.metric(
                f"{src.title()} Macro-F1",
                f"{metrics['test_macro_f1'][src]:.3f}",
                delta=f"acc {metrics['test_accuracy'][src]:.3f}",
            )
        w = metrics["fusion_weights"]
        st.caption(
            f"Fusion weights — text {w['text']:.0%} · meta {w['meta']:.0%}"
        )
else:
    st.warning("Run `python scripts/build_catalog.py && python scripts/train.py` before launching.")

st.sidebar.success("Select a page above to begin.")
