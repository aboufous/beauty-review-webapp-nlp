"""Task 1 — Browse and fuzzy keyword search."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.search import search  # noqa: E402
from src.ui import cached_products, render_product_card  # noqa: E402

st.set_page_config(page_title="Browse — GlowMate", page_icon="🛍", layout="wide")

st.title("🛍 Browse the catalogue")
st.caption(
    "Search by **brand**, **product name**, or **tag**. "
    "Fuzzy matching tolerates typos and word-order variations — "
    'try "maybeline" or "maybeline new york".'
)

products = cached_products()

with st.sidebar:
    st.header("Filters")
    query = st.text_input("Keyword search", value=st.session_state.get("query", ""))
    brands = ["(any brand)"] + sorted(products["brand_name"].dropna().unique().tolist())
    brand = st.selectbox("Brand", brands)
    min_rating = st.slider("Min avg rating", 0.0, 5.0, 0.0, step=0.5)
    max_results = st.slider("Max results", 6, 60, 18, step=6)

filtered = products.copy()
if brand != "(any brand)":
    filtered = filtered[filtered["brand_name"] == brand]
filtered = filtered[filtered["avg_product_rating"].fillna(0) >= min_rating]

results = search(query, filtered, limit=max_results) if query else filtered.head(max_results).assign(_score=None)

if query:
    st.success(f"Found {len(results):,} products matching “{query}”.")
else:
    st.info(f"Showing top {len(results)} products. Type a keyword to search.")

if len(results) == 0:
    st.warning("No products matched. Try a shorter or differently-spelled keyword.")

cols_per_row = 3
rows = (len(results) + cols_per_row - 1) // cols_per_row
for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        i = r * cols_per_row + c
        if i >= len(results):
            break
        with cols[c]:
            row = results.iloc[i].to_dict()
            render_product_card(row, score=row.get("_score"), key_prefix="browse")
