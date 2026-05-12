"""Task 1 fuzzy search over the product catalogue.

Builds a per-product search string from `brand_name + product_title + product_tags`
and ranks with rapidfuzz.WRatio so word-order variations
("Maybeline" vs "maybeline New York") collapse to the same score band.
"""
from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process


def _searchable(row) -> str:
    parts = [
        str(row.get("brand_name") or ""),
        str(row.get("product_title") or ""),
        str(row.get("product_tags") or ""),
    ]
    return " ".join(p for p in parts if p).lower()


def search(query: str, products: pd.DataFrame, limit: int = 60, threshold: int = 55) -> pd.DataFrame:
    if not query or not query.strip():
        return products.copy()

    choices = {idx: _searchable(row) for idx, row in products.iterrows()}
    matches = process.extract(
        query.lower().strip(),
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
    )
    hits = [(idx, score) for _, score, idx in matches if score >= threshold]
    if not hits:
        return products.iloc[0:0].assign(_score=[])

    idx_order = [i for i, _ in hits]
    scores = {i: s for i, s in hits}
    out = products.loc[idx_order].copy()
    out["_score"] = [scores[i] for i in idx_order]
    return out.reset_index(drop=True)
