"""Single integration point for loading the trained fused predictor.

Pages and modules import ``load_predictor()`` from here — never load joblibs
directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from scipy import sparse

from src.models import meta_model, sentiment_model, task3_best_model, text_model
from src.models.fusion import FusionModel
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

TEXT_PIPELINE_PATH = MODELS_DIR / "text_pipeline.joblib"
META_BUNDLE_PATH = MODELS_DIR / "meta_bundle.joblib"
FUSION_PATH = MODELS_DIR / "fusion.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
PRODUCT_TFIDF_PATH = MODELS_DIR / "product_tfidf_matrix.npz"
PRODUCT_INDEX_PATH = MODELS_DIR / "product_id_index.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
SENTIMENT_PATH = MODELS_DIR / "sentiment_model.joblib"
TASK3_BEST_PATH = MODELS_DIR / "task3_best_buyer_model.joblib"


@dataclass
class Prediction:
    proba: float
    label: bool
    proba_text: float
    proba_meta: float
    weight_text: float
    weight_meta: float
    model_name: str = "Fusion stacker"


class FusedPredictor:
    def __init__(self):
        self.text_pipeline = joblib.load(TEXT_PIPELINE_PATH)
        meta_bundle = joblib.load(META_BUNDLE_PATH)
        self.meta_pipeline = meta_bundle["pipeline"]
        self.top_brands = meta_bundle.get("top_brands", [])
        self.fusion: FusionModel = joblib.load(FUSION_PATH)

    def predict(self, review: dict, product: dict) -> Prediction:
        """review: title, review_text, review_rating
        product: brand_name, price, avg_product_rating, product_rating_count,
                 product_tags, product_id  (only used for compatibility — meta
                 model no longer reads product-level fields)
        """
        title = review.get("review_title", "") or ""
        body = review.get("review_text", "") or ""

        # No-signal guard. ``is_a_buyer`` in the dataset is base-rate-heavy
        # (78.7% True) and weakly tied to content, so on empty / gibberish
        # input both base models default to the prior and the fused score
        # stays near 0.8. That makes "asdf qwerty" read as a verified buyer,
        # which is nonsense from a UX standpoint. Detect no-vocab input at the
        # boundary and short-circuit to a low-confidence non-buyer verdict.
        cleaned = clean_text(f"{title} {body}").strip()
        tfidf = self.text_pipeline.named_steps["tfidf"]
        no_text_signal = (not cleaned) or tfidf.transform([cleaned]).nnz == 0

        p_text = text_model.predict_proba_one(self.text_pipeline, title, body)
        meta_row = {
            "review_rating": review.get("review_rating"),
            "review_title": title,
            "review_text": body,
        }
        p_meta = meta_model.predict_proba_one(self.meta_pipeline, self.top_brands, meta_row)
        p_fused = self.fusion.predict_proba(p_text, p_meta)

        if no_text_signal:
            p_fused = 0.2  # honest "we have no language signal to back this up"

        return Prediction(
            proba=p_fused,
            label=p_fused > self.fusion.decision_threshold,
            proba_text=p_text,
            proba_meta=p_meta,
            weight_text=self.fusion.weight_text,
            weight_meta=self.fusion.weight_meta,
            model_name="Fusion stacker",
        )


class Task3BestPredictor:
    def __init__(self):
        self.model: task3_best_model.Task3BestBuyerModel = joblib.load(TASK3_BEST_PATH)

    def predict(self, review: dict, product: dict) -> Prediction:
        p = self.model.predict_proba_one(review, product)
        return Prediction(
            proba=p,
            label=p > self.model.decision_threshold,
            proba_text=p,
            proba_meta=p,
            weight_text=1.0,
            weight_meta=0.0,
            model_name="Task 3 best: TF-IDF(1,2)+metadata LR",
        )


_PREDICTOR: FusedPredictor | None = None
_SENTIMENT_MODEL: sentiment_model.SentimentModel | None = None


def load_predictor() -> FusedPredictor:
    """Lazy singleton for the final HD fusion predictor."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = FusedPredictor()
    return _PREDICTOR


def load_sentiment_model() -> sentiment_model.SentimentModel:
    """Lazy singleton for customer sentiment prediction."""
    global _SENTIMENT_MODEL
    if _SENTIMENT_MODEL is None:
        _SENTIMENT_MODEL = joblib.load(SENTIMENT_PATH)
    return _SENTIMENT_MODEL


def load_product_tfidf():
    """(matrix, [product_id...]) for Task 3 similarity."""
    import json

    mat = sparse.load_npz(PRODUCT_TFIDF_PATH)
    ids = json.loads(PRODUCT_INDEX_PATH.read_text())
    return mat, ids
