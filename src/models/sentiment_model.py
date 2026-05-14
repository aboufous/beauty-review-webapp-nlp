"""Customer sentiment model for review text + rating.

This is separate from the buyer-authenticity model. The main fused classifier
predicts whether a review reads like it came from a verified buyer; this model
predicts whether the customer experience is positive or negative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocess import clean_text


@dataclass
class SentimentModel:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression

    def _features(self, texts: list[str], ratings: list[float]) -> sparse.csr_matrix:
        text_mat = self.vectorizer.transform([clean_text(t) for t in texts])
        rating_arr = np.asarray(ratings, dtype=np.float32).reshape(-1, 1)
        rating_arr = np.nan_to_num(rating_arr, nan=3.0)
        rating_scaled = (np.clip(rating_arr, 1.0, 5.0) - 1.0) / 4.0
        return sparse.hstack([text_mat, sparse.csr_matrix(rating_scaled)], format="csr")

    def predict_proba_one(self, text: str, rating: int | float) -> float:
        """Return P(customer sentiment is positive)."""
        x = self._features([text or ""], [rating])
        proba = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        pos_idx = classes.index(True) if True in classes else classes.index(1)
        return float(proba[pos_idx])

    def label_one(self, text: str, rating: int | float) -> tuple[str, float, str]:
        p_pos = self.predict_proba_one(text, rating)
        if p_pos >= 0.60:
            return "Positive", p_pos, "Good candidate for testimonial or recommendation"
        if p_pos <= 0.40:
            return "Negative", 1.0 - p_pos, "Customer recovery / product-quality follow-up"
        return "Mixed / neutral", max(p_pos, 1.0 - p_pos), "Monitor"


def train(reviews: pd.DataFrame) -> SentimentModel:
    """Fit a binary sentiment model from rating-derived labels.

    Ratings 4-5 are treated as positive, 1-2 as negative, and rating 3 is
    excluded from training because it is ambiguous.
    """
    df = reviews.copy()
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df = df.dropna(subset=["review_text", "review_rating"])
    df = df[df["review_rating"].isin([1, 2, 4, 5])].copy()
    y = df["review_rating"].ge(4).to_numpy()
    docs = (df["review_title"].fillna("") + " " + df["review_text"].fillna("")).map(clean_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
    )
    text_mat = vectorizer.fit_transform(docs)
    rating_scaled = ((df["review_rating"].to_numpy(dtype=np.float32).reshape(-1, 1) - 1.0) / 4.0)
    x = sparse.hstack([text_mat, sparse.csr_matrix(rating_scaled)], format="csr")
    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
    )
    clf.fit(x, y)
    return SentimentModel(vectorizer=vectorizer, clf=clf)
