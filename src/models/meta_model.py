"""Metadata classifier on rating + price + brand + product stats.

Strictly tabular features — no text content. Calibrated for fusion.
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "review_rating",
    "price",
    "avg_product_rating",
    "product_rating_count",
    "review_text_len",
    "review_title_len",
    "n_tags",
]
CATEGORICAL_FEATURES = ["brand_bucket"]
TOP_K_BRANDS = 25


def derive_features(df: pd.DataFrame, top_brands: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Pull / compute the meta features. Returns (X, top_brands)."""
    out = pd.DataFrame(index=df.index)
    out["review_rating"] = pd.to_numeric(df.get("review_rating"), errors="coerce")
    out["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    out["avg_product_rating"] = pd.to_numeric(df.get("avg_product_rating"), errors="coerce")
    out["product_rating_count"] = pd.to_numeric(df.get("product_rating_count"), errors="coerce")
    out["review_text_len"] = df.get("review_text", "").fillna("").astype(str).str.len()
    out["review_title_len"] = df.get("review_title", "").fillna("").astype(str).str.len()
    out["n_tags"] = (
        df.get("product_tags", "").fillna("").astype(str).str.count(",") + 1
    ).where(df.get("product_tags", "").fillna("") != "", 0)

    if top_brands is None:
        top_brands = (
            df["brand_name"].fillna("__missing__").value_counts().head(TOP_K_BRANDS).index.tolist()
        )
    brand = df["brand_name"].fillna("__missing__")
    out["brand_bucket"] = brand.where(brand.isin(top_brands), "__other__")
    return out, top_brands


def build_pipeline() -> Pipeline:
    """Calibrated HistGradientBoosting on numeric + one-hot brand."""
    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    base = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("features", preproc),
            ("clf", CalibratedClassifierCV(base, method="isotonic", cv=3)),
        ]
    )


def predict_proba_one(pipeline: Pipeline, top_brands: list[str], row: dict) -> float:
    df = pd.DataFrame([row])
    X, _ = derive_features(df, top_brands=top_brands)
    proba = pipeline.predict_proba(X)[0]
    classes = list(pipeline.classes_)
    pos_idx = classes.index(True) if True in classes else classes.index(1)
    return float(proba[pos_idx])
