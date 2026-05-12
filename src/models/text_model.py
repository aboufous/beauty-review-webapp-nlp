"""Text classifier: TF-IDF(1,2) + LogisticRegression — M1 grid-best."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text


def build_pipeline() -> Pipeline:
    """Return an unfitted text-classification pipeline.

    Hyperparameters reproduce the M1 GridSearchCV winner:
      TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.95)
      LogisticRegression(C=2.0, class_weight='balanced',
                         solver='liblinear', max_iter=2000)
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                ),
            ),
        ]
    )


def review_corpus(df) -> list[str]:
    """Concatenate cleaned title + body for each review row."""
    titles = df.get("review_title", "").fillna("").astype(str)
    bodies = df.get("review_text", "").fillna("").astype(str)
    return [clean_text(f"{t} {b}") for t, b in zip(titles, bodies)]


def predict_proba_one(pipeline: Pipeline, title: str, body: str) -> float:
    """P(is_a_buyer=True) for a single review."""
    text = clean_text(f"{title or ''} {body or ''}")
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    # Handle both boolean and int class labels.
    pos_idx = classes.index(True) if True in classes else classes.index(1)
    return float(proba[pos_idx])
