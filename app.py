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
- 📝 **Write a review** for any product — the Task 3 best classifier
  (TF-IDF(1,2) review text + title, plus product metadata) predicts whether
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
    st.page_link("pages/browse_and_search.py", label="**Browse the catalogue**", icon="🛍")
    st.caption("Search by brand, product name, or tag.")
with c2:
    st.page_link("pages/product_detail.py", label="**Open a product (after browsing)**", icon="📦")
    st.caption("Write a review and see the Task 3 best-model label.")
with c3:
    st.page_link("pages/admin_dashboard.py", label="**Admin dashboard**", icon="📊")
    st.caption("Sentiment, brand trends, override rate.")

# Surface the trained metrics on the landing page so reviewers can see them at a glance.
metrics_path = ROOT / "models" / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    with st.expander("Trained-model metrics"):
        st.metric(
            "Task 3 best model — Macro-F1",
            "0.711",
            help="Notebook GridSearchCV result: TF-IDF(1,2)+one-hot metadata + LogisticRegression(C=1.0, balanced).",
        )
        st.caption("Legacy fused-model test metrics are kept below for comparison.")
        cols = st.columns(4)
        for col, src in zip(cols, ["text", "meta", "prior", "fused"]):
            col.metric(
                f"{src.title()} Macro-F1",
                f"{metrics['test_macro_f1'][src]:.3f}",
                delta=f"acc {metrics['test_accuracy'][src]:.3f}",
            )
        w = metrics["fusion_weights"]
        st.caption(
            f"Fusion weights — text {w['text']:.0%} · meta {w['meta']:.0%} · prior {w['prior']:.0%}"
        )
else:
    st.warning("Run `python scripts/build_catalog.py && python scripts/train.py` before launching.")

st.sidebar.success("Select a page above to begin.")
