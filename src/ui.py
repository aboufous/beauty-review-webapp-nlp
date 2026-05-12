"""Streamlit-side helpers shared by pages."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_products, load_reviews


def placeholder_image(product_id) -> str:
    """Stable Picsum URL per product_id — satisfies the brief's
    "artificial images for display" allowance."""
    h = int(hashlib.md5(str(product_id).encode()).hexdigest()[:6], 16) % 1000
    return f"https://picsum.photos/seed/glowmate-{h}/320/220"


@st.cache_data(ttl=60)
def cached_products() -> pd.DataFrame:
    return load_products()


@st.cache_data(ttl=15)  # short TTL so newly-submitted reviews appear fast
def cached_reviews() -> pd.DataFrame:
    return load_reviews()


def rating_stars(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    full = int(round(float(value)))
    return "★" * full + "☆" * (5 - full)


def render_product_card(row: dict, *, score: float | None = None, key_prefix: str = ""):
    pid = row.get("product_id")
    with st.container(border=True):
        st.image(placeholder_image(pid), use_column_width=True)
        st.markdown(f"**{row.get('product_title') or '(no title)'}**")
        st.caption(row.get("brand_name") or "—")
        cols = st.columns(2)
        cols[0].write(rating_stars(row.get("avg_product_rating")))
        price = row.get("price")
        cols[1].write(f"₹{int(price):,}" if pd.notna(price) else "—")
        if score is not None:
            st.caption(f"match score: {score:.0f}" if score > 1 else f"similarity: {score:.2f}")
        if st.button("View details", key=f"{key_prefix}view-{pid}", use_container_width=True):
            # st.switch_page does not preserve query_params set in the same run,
            # so we hand the selection over via session_state and let Page 2
            # write it into the URL on arrival.
            st.session_state["selected_product_id"] = str(pid)
            st.switch_page("pages/2_📦_Product_Detail.py")
