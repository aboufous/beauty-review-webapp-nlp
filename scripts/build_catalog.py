"""Materialise data/products.csv and data/reviews.csv from the M1 raw CSV.

Run once before the webapp:

    python scripts/build_catalog.py

Idempotent — re-running overwrites the files.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (DATA_DIR, PRODUCTS_CSV, REVIEWS_CSV,  # noqa: E402
                      USER_REVIEW_COLUMNS, build_products, load_raw_reviews)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw M1 reviews …")
    raw = load_raw_reviews()
    print(f"  raw shape: {raw.shape}")

    print("Building products.csv …")
    products = build_products(raw)
    products.to_csv(PRODUCTS_CSV, index=False)
    print(f"  wrote {PRODUCTS_CSV.relative_to(ROOT)} — {len(products):,} products")

    print("Seeding reviews.csv …")
    seed = raw.copy()
    for col in USER_REVIEW_COLUMNS:
        if col not in seed.columns:
            seed[col] = None
    seed = seed[USER_REVIEW_COLUMNS]
    seed.to_csv(REVIEWS_CSV, index=False)
    print(f"  wrote {REVIEWS_CSV.relative_to(ROOT)} — {len(seed):,} reviews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
