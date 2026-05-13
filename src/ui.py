"""Streamlit-side helpers shared by pages."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from src.data import load_products, load_reviews


# Ordered list of (keyword-pattern, Flickr-tag) — first match wins.
# Most specific patterns first so "lip crayon" beats a bare "lip".
_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("lipstick", "lipstick"),
    ("lip crayon", "lipstick"),
    ("lip cream", "lipstick"),
    ("lip color", "lipstick"),
    ("lip liner", "lipstick"),
    ("lip gloss", "lipgloss"),
    ("lip balm", "lipbalm"),
    ("eyeliner", "eyeliner"),
    ("kohl", "eyeliner"),
    ("kajal", "eyeliner"),
    ("mascara", "mascara"),
    ("eyeshadow", "eyeshadow"),
    ("eye shadow", "eyeshadow"),
    ("eye palette", "eyeshadow"),
    ("foundation", "foundation"),
    ("concealer", "concealer"),
    ("primer", "makeup"),
    ("blush", "blush"),
    ("highlighter", "highlighter"),
    ("compact", "powder"),
    ("face powder", "powder"),
    ("loose powder", "powder"),
    ("setting powder", "powder"),
    ("contour", "makeup"),
    ("brow", "eyebrow"),
    ("nail enamel", "nailpolish"),
    ("nail polish", "nailpolish"),
    ("nail color", "nailpolish"),
    ("shampoo", "shampoo"),
    ("conditioner", "conditioner"),
    ("hair color", "haircolor"),
    ("hair oil", "hairoil"),
    ("face wash", "facewash"),
    ("cleanser", "facewash"),
    ("face mask", "facemask"),
    ("clay mask", "facemask"),
    ("sheet mask", "facemask"),
    ("serum", "serum"),
    ("moisturizer", "moisturizer"),
    ("day cream", "skincare"),
    ("night cream", "skincare"),
    ("eye cream", "skincare"),
    ("sunscreen", "sunscreen"),
    ("body lotion", "bodylotion"),
    ("body wash", "showergel"),
    ("shower gel", "showergel"),
    ("perfume", "perfume"),
    ("fragrance", "perfume"),
    ("toner", "skincare"),
    ("scrub", "skincare"),
    ("makeup", "makeup"),
]


def _category_tag(title: str | None) -> str:
    """Pick a Flickr tag for `title` by scanning for known cosmetics keywords."""
    if not title or pd.isna(title):
        return "cosmetics"
    t = str(title).lower()
    for needle, tag in _CATEGORY_PATTERNS:
        if needle in t:
            return tag
    return "cosmetics"


def product_image(product_id, title: str | None = None) -> str:
    """Stock photo matching the product category, deterministic per product_id.

    We scan the product title for a category keyword (lipstick, mascara,
    shampoo, …) and ask Loremflickr for a Flickr photo tagged with that
    category. The `lock` query parameter pins the image so the same product
    always shows the same photo. Only a generic category keyword is sent
    over the wire — brand/title strings are not.
    """
    seed = int(hashlib.md5(str(product_id).encode()).hexdigest()[:8], 16) % 1_000_000
    tag = _category_tag(title)
    return f"https://loremflickr.com/320/220/{quote(tag)},cosmetics/all?lock={seed}"


# Back-compat alias for any callers still importing the old name.
placeholder_image = product_image


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
        st.image(
            product_image(pid, title=row.get("product_title")),
            use_column_width=True,
        )
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
