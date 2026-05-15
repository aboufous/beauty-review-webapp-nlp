"""Metadata classifier — review-level features only.

Earlier versions of this model mixed in product-level features (price,
avg_product_rating, product_rating_count, brand_bucket, n_tags). Those leaked
product identity into the meta signal: on a popular product every review got
the same proba near 1.0, regardless of content (see task2_report.html
"Hạn chế..."). To force the meta channel to react to the review itself, the
feature set is now restricted to attributes derived from the review row:

    - review_rating
    - review_text_len
    - review_title_len

Same Calibrated HistGradientBoosting backbone — only the inputs change.
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "review_rating",
    "review_text_len",
    "review_title_len",
]
CATEGORICAL_FEATURES: list[str] = []


def derive_features(df: pd.DataFrame, top_brands: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Pull / compute the review-level meta features.

    The ``top_brands`` argument is kept for backwards compatibility with the
    training script and loader bundle — it is no longer used. Returns an empty
    list as the second element.
    """
    out = pd.DataFrame(index=df.index)
    out["review_rating"] = pd.to_numeric(df.get("review_rating"), errors="coerce")
    out["review_text_len"] = df.get("review_text", "").fillna("").astype(str).str.len()
    out["review_title_len"] = df.get("review_title", "").fillna("").astype(str).str.len()
    return out, []


def build_pipeline() -> Pipeline:
    """Calibrated HistGradientBoosting on review-level numeric features."""
    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
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
