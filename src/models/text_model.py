"""Text classifier: TF-IDF(1,2) + LogisticRegression — M1 grid-best."""
from __future__ import annotations

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
    """P(is_a_buyer=True) for a single review.

    Returns 0.5 (no-info) when the cleaned text has zero in-vocabulary tokens —
    i.e. empty input, pure gibberish, or text whose only words were filtered by
    ``min_df``. Without this guard the LR falls back to its intercept, which is
    biased toward the majority class (78.7% buyer) and would label every
    gibberish review as a buyer.
    """
    text = clean_text(f"{title or ''} {body or ''}")
    if not text.strip():
        return 0.5
    tfidf = pipeline.named_steps["tfidf"]
    vec = tfidf.transform([text])
    if vec.nnz == 0:
        return 0.5
    proba = pipeline.named_steps["clf"].predict_proba(vec)[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    pos_idx = classes.index(True) if True in classes else classes.index(1)
    return float(proba[pos_idx])
