"""Streamlit-side helpers shared by pages."""
from __future__ import annotations

import hashlib
from urllib.parse import quote

import pandas as pd
import streamlit as st

from src.data import load_products, load_reviews

# Per category: (display label, emoji glyph, gradient start, gradient end).
# Used to render an SVG card that visually represents the product without
# showing any people / faces. Order matters — first match wins.
_CATEGORY_META: list[tuple[str, str, str, str, str]] = [
    # needle,           label,        emoji,  color1,     color2
    ("lip crayon",      "Lip crayon", "💄",   "#ffd5e5",  "#ff6b9d"),
    ("lip cream",       "Lip cream",  "💄",   "#ffd5e5",  "#ff6b9d"),
    ("lip color",       "Lip color",  "💄",   "#ffd5e5",  "#ff6b9d"),
    ("lip liner",       "Lip liner",  "💄",   "#ffd5e5",  "#ff6b9d"),
    ("lip gloss",       "Lip gloss",  "✨",   "#ffe4f0",  "#f8a5c2"),
    ("lip balm",        "Lip balm",   "💋",   "#fff0f3",  "#ffa5b5"),
    ("lipstick",        "Lipstick",   "💄",   "#ffd5e5",  "#ff4d7d"),
    ("eyeliner",        "Eyeliner",   "👁",   "#3a3a3a",  "#0d0d0d"),
    ("kohl",            "Kohl",       "👁",   "#3a3a3a",  "#0d0d0d"),
    ("kajal",           "Kajal",      "👁",   "#3a3a3a",  "#0d0d0d"),
    ("mascara",         "Mascara",    "🖤",   "#2a2a2a",  "#1a1a3a"),
    ("eyeshadow",       "Eyeshadow",  "🎨",   "#d4a5a5",  "#8e5a8e"),
    ("eye shadow",      "Eyeshadow",  "🎨",   "#d4a5a5",  "#8e5a8e"),
    ("eye palette",     "Palette",    "🎨",   "#d4a5a5",  "#8e5a8e"),
    ("eye cream",       "Eye cream",  "🌿",   "#f5e6d3",  "#c9b29a"),
    ("foundation",      "Foundation", "🪞",   "#f5dcc4",  "#c9a482"),
    ("concealer",       "Concealer",  "🪞",   "#f5dcc4",  "#c9a482"),
    ("primer",          "Primer",     "✨",   "#f0e4d7",  "#b8a288"),
    ("blush",           "Blush",      "🌸",   "#ffdada",  "#ff8aa0"),
    ("highlighter",     "Highlighter","✨",   "#fff5dc",  "#f0c75a"),
    ("loose powder",    "Powder",     "🌼",   "#fbe9c8",  "#d9b676"),
    ("setting powder",  "Powder",     "🌼",   "#fbe9c8",  "#d9b676"),
    ("face powder",     "Powder",     "🌼",   "#fbe9c8",  "#d9b676"),
    ("compact",         "Compact",    "🌼",   "#fbe9c8",  "#d9b676"),
    ("contour",         "Contour",    "🎯",   "#e8c8a5",  "#a8794c"),
    ("brow",            "Eyebrow",    "✏️",  "#d4b896",  "#8c6a40"),
    ("nail enamel",     "Nail polish","💅",   "#e8c5ff",  "#a059e0"),
    ("nail polish",     "Nail polish","💅",   "#e8c5ff",  "#a059e0"),
    ("nail color",      "Nail polish","💅",   "#e8c5ff",  "#a059e0"),
    ("shampoo",         "Shampoo",    "🧴",   "#cfe9ff",  "#5aa0d8"),
    ("conditioner",     "Conditioner","🧴",   "#d4f0e8",  "#5fb3a1"),
    ("hair color",      "Hair color", "🎨",   "#d8b5a5",  "#8b4a2f"),
    ("hair oil",        "Hair oil",   "🌿",   "#f5e6c0",  "#c79a3a"),
    ("face wash",       "Face wash",  "🧼",   "#dff3ff",  "#82c5e8"),
    ("cleanser",        "Cleanser",   "🧼",   "#dff3ff",  "#82c5e8"),
    ("face mask",       "Face mask",  "🌿",   "#d4ecdc",  "#5fa776"),
    ("clay mask",       "Clay mask",  "🌿",   "#e8d4b8",  "#a07a4a"),
    ("sheet mask",      "Sheet mask", "🌿",   "#d4ecdc",  "#5fa776"),
    ("serum",           "Serum",      "💧",   "#d8eef5",  "#5a9bc4"),
    ("moisturizer",     "Moisturiser","💧",   "#e3f2e8",  "#7ab896"),
    ("day cream",       "Day cream",  "☀️",  "#fff0c8",  "#e8b850"),
    ("night cream",     "Night cream","🌙",   "#dcd2f0",  "#6a5acd"),
    ("sunscreen",       "Sunscreen",  "☀️",  "#fff0a8",  "#e8a420"),
    ("body lotion",     "Body lotion","🌿",   "#e8dcc8",  "#a8855a"),
    ("body wash",       "Body wash",  "🫧",   "#dff3ff",  "#5fa0d4"),
    ("shower gel",      "Shower gel", "🫧",   "#dff3ff",  "#5fa0d4"),
    ("perfume",         "Perfume",    "🌸",   "#f0dfff",  "#b078e0"),
    ("fragrance",       "Fragrance",  "🌸",   "#f0dfff",  "#b078e0"),
    ("toner",           "Toner",      "💧",   "#dff0f5",  "#6aaec0"),
    ("scrub",           "Scrub",      "🌿",   "#e8dcc8",  "#a8855a"),
    ("makeup",          "Makeup",     "🎨",   "#ffdfd0",  "#e89570"),
]

_DEFAULT_META = ("Cosmetics", "✨", "#ffe0e9", "#d96a8a")


def _category_meta(title: str | None) -> tuple[str, str, str, str]:
    if title and pd.notna(title):
        t = str(title).lower()
        for needle, label, emoji, c1, c2 in _CATEGORY_META:
            if needle in t:
                return label, emoji, c1, c2
    return _DEFAULT_META


def product_image(product_id, title: str | None = None) -> str:
    """Deterministic SVG category card for the product.

    Generated inline from the category extracted from ``title`` — gradient
    + emoji + label. No external requests, no real-photo dependency.
    The brief allows artificial display images, so an SVG placeholder is
    compliant for every product.
    """
    label, emoji, c1, c2 = _category_meta(title)
    seed = int(hashlib.md5(str(product_id).encode()).hexdigest()[:8], 16)
    angle = seed % 360
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="220" viewBox="0 0 320 220">'
        f'<defs><linearGradient id="g" gradientTransform="rotate({angle} 0.5 0.5)">'
        f'<stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/>'
        '</linearGradient></defs>'
        '<rect width="320" height="220" fill="url(#g)" rx="10"/>'
        '<text x="160" y="125" font-size="84" text-anchor="middle" dominant-baseline="middle" '
        'font-family="Apple Color Emoji, Segoe UI Emoji, sans-serif">'
        f'{emoji}</text>'
        '<text x="160" y="190" font-size="14" text-anchor="middle" '
        'font-family="-apple-system, Segoe UI, sans-serif" font-weight="600" '
        f'fill="rgba(0,0,0,0.65)" letter-spacing="1.5">{label.upper()}</text>'
        '</svg>'
    )
    return f"data:image/svg+xml;utf8,{quote(svg)}"


# Back-compat alias.
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
            st.switch_page("pages/Product_Detail.py")
