"""Task 1 – Fuzzy search over the product catalogue.

Builds a per-product search string from brand_name + product_title + product_tags,
normalises both the user query and the product text (brand aliases, stop words),
then ranks matches using RapidFuzz WRatio.

Filtering pipeline (4 layers):
  1. Brand-aware check  – if the query contains a known brand, only products of
                          that brand survive (fixes "maybelline mascara" → Lakme)
  2. Token-level filter – every remaining query token must fuzzy-match ≥1 product token
  3. Global WRatio rank – orders survivors by overall similarity score
  4. Min-score cutoff   – drops anything below min_score (default 60)
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Brand aliases – longest first to avoid partial-replacement bugs
# ---------------------------------------------------------------------------
BRAND_ALIASES: list[tuple[str, str]] = sorted(
    [
        ("maybelline new york", "maybelline"),
        ("maybe line new york", "maybelline"),
        ("maybe line",          "maybelline"),
        ("maybeline new york",  "maybelline"),
        ("maybeline",           "maybelline"),
        ("l'oreal paris",       "loreal"),
        ("loreal paris",        "loreal"),
        ("l oreal paris",       "loreal"),
        ("l oreal",             "loreal"),
        # Add more pairs here as needed
    ],
    key=lambda x: -len(x[0]),
)

# Canonical brand names (normalised) known in the catalogue.
# Used to detect when the user's query names a specific brand.
KNOWN_BRANDS: set[str] = {
    "maybelline", "loreal", "olay", "nykaa", "lakme", "mac",
    "revlon", "NYX", "cetaphil", "neutrogena", "garnier", "dove",
    # Add every brand in your catalogue here
}

STOP_WORDS = {
    "new", "york", "paris", "london", "cosmetics", "usa",
    "the", "of", "and", "inc", "llc", "group", "collection",
}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, apply brand aliases, remove stop words."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)   # remove punctuation
    t = re.sub(r"\s+", " ", t).strip()    # collapse spaces

    for alias, target in BRAND_ALIASES:    # longest-first replacement
        t = t.replace(alias, target)

    tokens = [tok for tok in t.split() if tok not in STOP_WORDS]
    return " ".join(tokens)


def _normalize_brand(brand: str) -> str:
    """Normalise a raw brand_name cell (same pipeline as _normalize)."""
    return _normalize(brand)


def _searchable(row: pd.Series) -> str:
    """Concatenate brand + title + tags into one normalised search string."""
    parts = []
    for col in ("brand_name", "product_title", "product_tags"):
        val = row.get(col)
        if pd.notna(val):
            s = str(val).strip()
            if s.lower() not in ("", "nan", "none"):
                s = re.sub(r"[,|;]+", " ", s)   # tags: replace separators
                parts.append(s)
    return _normalize(" ".join(parts))


def _extract_brand_from_query(query_norm: str) -> str | None:
    """Return the canonical brand name if the query contains a known brand,
    else None.  Checks single tokens and two-token combos."""
    tokens = query_norm.split()
    # Single token check
    for tok in tokens:
        if tok in KNOWN_BRANDS:
            return tok
    # Two-token combo (e.g. "nykaa cosmetics" → "nykaa" already handled by alias,
    # but keeps this as safety net)
    for i in range(len(tokens) - 1):
        combo = tokens[i] + " " + tokens[i + 1]
        if combo in KNOWN_BRANDS:
            return combo
    return None


def _brand_matches(product_brand_norm: str, target_brand: str) -> bool:
    """True if the product's normalised brand fuzzy-matches the target brand."""
    return fuzz.ratio(product_brand_norm, target_brand) >= 80


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search(
    query: str,
    products: pd.DataFrame,
    limit: int = 60,
    token_match_threshold: int = 80,
    min_score: int = 60,
) -> pd.DataFrame:
    """Search the product catalogue and return ranked, filtered results.

    Parameters
    ----------
    query               : raw user input
    products            : DataFrame with brand_name, product_title, product_tags
    limit               : max results to return
    token_match_threshold: min fuzz.ratio for a query token to count as matched
    min_score           : min WRatio score to appear in results (cuts noise)

    Returns
    -------
    DataFrame with same columns as products plus _score, sorted desc by score.
    Empty DataFrame if nothing matches.
    """
    if not query or not query.strip():
        return products.copy()

    query_norm = _normalize(query)
    if not query_norm:
        return products.copy()

    query_tokens = query_norm.split()

    # ------------------------------------------------------------------
    # Step 1 – Brand-aware pre-filter
    # If the query names a known brand, keep only products of that brand.
    # This is what stops "maybelline mascara" from returning Lakme products.
    # ------------------------------------------------------------------
    detected_brand = _extract_brand_from_query(query_norm)

    if detected_brand:
        brand_filtered = products[
            products["brand_name"].apply(
                lambda b: _brand_matches(_normalize_brand(str(b)), detected_brand)
            )
        ]
        # If brand filter leaves nothing (brand not in catalogue), fall back
        candidate_pool = brand_filtered if not brand_filtered.empty else products
    else:
        candidate_pool = products

    # ------------------------------------------------------------------
    # Step 2 – Token-level fuzzy filter
    # Every query token must fuzzy-match at least one token in the product text.
    # ------------------------------------------------------------------
    # Remove the brand token(s) from query_tokens before this check so we don't
    # double-penalise for brand (it was already handled in Step 1).
    non_brand_tokens = [t for t in query_tokens if t != detected_brand] if detected_brand else query_tokens

    choices: dict[int, str] = {}

    for idx, row in candidate_pool.iterrows():
        text = _searchable(row)
        if not text:
            continue

        product_tokens = text.split()

        # If there are no non-brand tokens (user typed only the brand name),
        # accept all brand-filtered products directly.
        if not non_brand_tokens:
            choices[idx] = text
            continue

        all_matched = True
        for qt in non_brand_tokens:
            best_score = max(
                (fuzz.ratio(qt, pt) for pt in product_tokens),
                default=0,
            )
            # Substring bonus for tokens ≥ 4 chars (avoids "a"/"in" matching everything)
            substring_bonus = (
                len(qt) >= 4
                and any(qt in pt or pt in qt for pt in product_tokens)
            )
            if best_score < token_match_threshold and not substring_bonus:
                all_matched = False
                break

        if all_matched:
            choices[idx] = text

    if not choices:
        return products.iloc[0:0].assign(_score=[])

    # ------------------------------------------------------------------
    # Step 3 – Global WRatio ranking among pre-filtered products
    # ------------------------------------------------------------------
    matches = process.extract(
        query_norm,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
    )
    hits = [(idx, score) for _, score, idx in matches]

    if not hits:
        return products.iloc[0:0].assign(_score=[])

    idx_order = [i for i, _ in hits]
    scores = {i: s for i, s in hits}

    out = products.loc[idx_order].copy()
    out["_score"] = [scores[i] for i in idx_order]

    # ------------------------------------------------------------------
    # Step 4 – Min-score cutoff (removes residual noise)
    # ------------------------------------------------------------------
    out = out[out["_score"] >= min_score]

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke-test  →  python search.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame([
        {"brand_name": "Maybelline New York", "product_title": "Fit Me Foundation",     "product_tags": "foundation, face, coverage"},
        {"brand_name": "Maybelline New York", "product_title": "Great Lash Mascara",    "product_tags": "mascara, eyes, lashes"},
        {"brand_name": "Lakme",               "product_title": "Absolute Illuminating Blush Shimmer + Eyeconic Curling Mascara Combo", "product_tags": "mascara, blush, eyes, combo"},
        {"brand_name": "L'Oreal Paris",       "product_title": "True Match Foundation", "product_tags": "foundation, face"},
        {"brand_name": "Olay",                "product_title": "Regenerist Micro-Sculpting Cream", "product_tags": "moisturizer, face, anti-aging"},
        {"brand_name": "Nykaa Cosmetics",     "product_title": "Wing In A Blink Eyeliner Pen",    "product_tags": "eyeliner, eyes"},
        {"brand_name": "MAC",                 "product_title": "Ruby Woo Lipstick",     "product_tags": "lipstick, lips, red"},
    ])

    test_cases = [
        # (query,                   expected_brands_only,          should_be_empty)
        ("maybelline mascara",      ["Maybelline New York"],        False),
        ("Maybeline",               ["Maybelline New York"],        False),
        ("maybe line New York",     ["Maybelline New York"],        False),
        ("Olay",                    ["Olay"],                       False),
        ("loreal foundation",       ["L'Oreal Paris"],              False),
        ("mascara",                 None,                           False),   # any brand with mascara
        ("xyzabc",                  None,                           True),    # must return nothing
        ("Olay lipstick",           None,                           True),    # Olay has no lipstick
    ]

    for query, expected_brands, expect_empty in test_cases:
        res = search(query, sample)
        status = "✅" if (res.empty == expect_empty) else "❌"
        if expected_brands and not res.empty:
            wrong = res[~res["brand_name"].isin(expected_brands)]
            status = "✅" if wrong.empty else f"❌ WRONG BRAND: {wrong['brand_name'].tolist()}"
        print(f"{status}  '{query}' → {len(res)} result(s)  scores={list(res['_score']) if not res.empty else []}")
        if not res.empty:
            for _, r in res.iterrows():
                print(f"     {r['brand_name']} | {r['product_title']} | score={r['_score']}")
                
