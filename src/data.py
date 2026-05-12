"""Data loading and persistence for the GlowMate webapp.

Two CSVs back the running app:

* ``data/products.csv``    — one row per ``product_id``, derived from the raw
  Milestone-I dataset by ``scripts/build_catalog.py``. Static at runtime.
* ``data/reviews.csv``     — full review corpus. Seeded once from M1's raw data
  and appended to whenever a user submits a review via Task 2.

The original M1 ``processed.csv`` retains only the tokenised text. The webapp
needs the *raw* review text for display, so we read from
``MAI_Group5/cosmetics_beauty_products_reviews.csv`` when materialising the
seed and let ``preprocess.clean_text`` re-derive features at inference time.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
REVIEWS_CSV = DATA_DIR / "reviews.csv"

# Raw M1 source. We treat MAI_Group5/ as read-only.
M1_RAW_CSV = ROOT / "MAI_Group5" / "cosmetics_beauty_products_reviews.csv"

RAW_COLUMNS = [
    "product_id",
    "brand_name",
    "review_id",
    "review_title",
    "review_text",
    "author",
    "review_date",
    "review_rating",
    "is_a_buyer",
    "product_title",
    "price",
    "avg_product_rating",
    "product_rating_count",
    "product_tags",
    "product_url",
]

# Columns added by the webapp when a user submits a review.
USER_REVIEW_COLUMNS = [
    *RAW_COLUMNS,
    "predicted_label",
    "predicted_proba",
    "user_overrode",
    "created_at",
]


def load_raw_reviews() -> pd.DataFrame:
    """Read the raw M1 review CSV. Cheap to cache — caller decides."""
    df = pd.read_csv(M1_RAW_CSV)
    df["is_a_buyer"] = df["is_a_buyer"].astype("boolean")
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["avg_product_rating"] = pd.to_numeric(df["avg_product_rating"], errors="coerce")
    df["product_rating_count"] = pd.to_numeric(df["product_rating_count"], errors="coerce")
    return df


def build_products(reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per product_id with aggregated metadata."""
    def _first(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if len(s) else None

    grouped = reviews.groupby("product_id", as_index=False).agg(
        brand_name=("brand_name", _first),
        product_title=("product_title", _first),
        product_tags=("product_tags", _first),
        product_url=("product_url", _first),
        price=("price", "median"),
        avg_product_rating=("avg_product_rating", "first"),
        product_rating_count=("product_rating_count", "first"),
        n_reviews=("review_id", "count"),
        mean_review_rating=("review_rating", "mean"),
        pct_is_a_buyer=("is_a_buyer", lambda s: s.astype("float").mean()),
    )
    return grouped.sort_values("n_reviews", ascending=False).reset_index(drop=True)


def load_products() -> pd.DataFrame:
    if not PRODUCTS_CSV.exists():
        raise FileNotFoundError(
            f"{PRODUCTS_CSV} missing — run `python scripts/build_catalog.py` first."
        )
    return pd.read_csv(PRODUCTS_CSV)


def load_reviews() -> pd.DataFrame:
    if not REVIEWS_CSV.exists():
        raise FileNotFoundError(
            f"{REVIEWS_CSV} missing — run `python scripts/build_catalog.py` first."
        )
    df = pd.read_csv(REVIEWS_CSV, low_memory=False)
    for col in USER_REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["is_a_buyer"] = df["is_a_buyer"].astype("boolean")
    return df


def reviews_for_product(product_id, reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    df = reviews if reviews is not None else load_reviews()
    out = df[df["product_id"].astype(str) == str(product_id)].copy()
    if "created_at" in out.columns:
        # Newest user-submitted reviews float to the top; M1 rows have NaT.
        out["_sort"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
        out = out.sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort")
    return out


def append_review(row: dict) -> str:
    """Append a user-submitted review to data/reviews.csv atomically.

    Returns the generated review_id.
    """
    review_id = row.get("review_id") or uuid.uuid4().hex[:12]
    row["review_id"] = review_id
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("review_date", row["created_at"])

    df = load_reviews() if REVIEWS_CSV.exists() else pd.DataFrame(columns=USER_REVIEW_COLUMNS)
    new_row = pd.DataFrame([{c: row.get(c) for c in USER_REVIEW_COLUMNS}])
    # Align dtypes before concat to silence pandas FutureWarning.
    for col in new_row.columns:
        if col in df.columns and df[col].dtype != "object":
            new_row[col] = new_row[col].astype(df[col].dtype, errors="ignore")
    df = pd.concat([df, new_row], ignore_index=True) if len(df) else new_row

    # Atomic write: temp file in same dir, then os.replace.
    fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, prefix=".reviews-", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, REVIEWS_CSV)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return review_id


def product_row(product_id, products: pd.DataFrame | None = None) -> dict | None:
    df = products if products is not None else load_products()
    hit = df[df["product_id"].astype(str) == str(product_id)]
    return hit.iloc[0].to_dict() if len(hit) else None
