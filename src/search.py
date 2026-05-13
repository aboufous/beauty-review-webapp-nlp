"""Task 1 fuzzy search over the product catalogue.

This module builds a per-product search string from `brand_name + product_title + product_tags`,
normalises both the user query and the product text (brand aliases, stop words),
then ranks matches using RapidFuzz `WRatio`. Normalisation ensures that spelling variations
such as 'Maybe line' and 'maybe line New York' yield exactly the same search results.
"""
from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process

# ----------------------------------------------------------------------
# Normalisation utilities – guarantee identical results for similar queries
# ----------------------------------------------------------------------

# Dictionary mapping common misspellings/variations to canonical brand names
BRAND_ALIASES = {
    "maybe line": "maybelline",
    "maybelline new york": "maybelline",
    "l'oreal paris": "loreal",
    "loreal paris": "loreal",
    # Add more entries as needed based on your dataset
}

# Stop words: words that do not contribute to brand/product identity
STOP_WORDS = {
    "new", "york", "paris", "london", "cosmetics", "usa",
    "the", "of", "and", "inc", "llc", "group", "collection"
}


def _normalize(text: str) -> str:
    """Clean and canonicalise a text string for search purposes.

    1. Lowercase
    2. Remove punctuation (keep alphanumeric and spaces)
    3. Replace known brand aliases with their canonical form
    4. Drop stop words

    Examples:
        "Maybe line"          -> "maybelline"
        "maybe line New York" -> "maybelline"
        "Olay cream"          -> "olay cream"
    """
    t = text.lower().strip()
    # Keep only alphanumeric characters and spaces
    t = ''.join(c for c in t if c.isalnum() or c.isspace())
    # Replace aliases
    for alias, target in BRAND_ALIASES.items():
        if alias in t:
            t = t.replace(alias, target)
    # Remove stop words
    tokens = [tok for tok in t.split() if tok not in STOP_WORDS]
    return " ".join(tokens)


def _searchable(row) -> str:
    """Build a normalised search string for a single product row."""
    parts = [
        str(row.get("brand_name") or ""),
        str(row.get("product_title") or ""),
        str(row.get("product_tags") or ""),
    ]
    raw = " ".join(p for p in parts if p)
    return _normalize(raw)


# ----------------------------------------------------------------------
# Main search function – called by the Streamlit page
# ----------------------------------------------------------------------
def search(
    query: str,
    products: pd.DataFrame,
    limit: int = 60,
    threshold: int = 55,
) -> pd.DataFrame:
    """Return products matching the query, ranked by fuzzy relevance score.

    Both the user query and the product search texts are normalised before
    comparison, so 'Maybe line' and 'maybe line New York' receive identical
    scores and therefore return the same product list.

    Parameters
    ----------
    query : str
        Raw user input (e.g. "maybe line new york").
    products : pd.DataFrame
        Product catalogue; must contain at least `brand_name`, `product_title`
        and optionally `product_tags`.
    limit : int
        Maximum number of results to return.
    threshold : int
        Minimum RapidFuzz score (0-100) for a match to be included.

    Returns
    -------
    pd.DataFrame
        Filtered and ranked products, with an added `_score` column (float).
    """
    # If no query is provided, return the original DataFrame unchanged
    if not query or not query.strip():
        return products.copy()

    # Normalise the user input
    query_norm = _normalize(query)

    # Build normalised strings for every product
    choices = {idx: _searchable(row) for idx, row in products.iterrows()}

    # Perform fuzzy extraction (fast batch scoring)
    matches = process.extract(
        query_norm,
        choices,
        scorer=fuzz.WRatio,   # Weighted ratio – robust to token order changes
        limit=limit,
    )
    hits = [(idx, score) for _, score, idx in matches if score >= threshold]

    # No match → return empty DataFrame with a _score column
    if not hits:
        return products.iloc[0:0].assign(_score=[])

    # Reorder products by match score (highest first)
    idx_order = [i for i, _ in hits]
    scores = {i: s for i, s in hits}
    out = products.loc[idx_order].copy()
    out["_score"] = [scores[i] for i in idx_order]
    return out.reset_index(drop=True)
