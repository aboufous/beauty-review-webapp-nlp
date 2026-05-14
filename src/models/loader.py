"""Single integration point for loading the trained fused predictor.

Pages and modules import ``load_predictor()`` from here — never load joblibs
directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from scipy import sparse

from src.models import meta_model, prior_model, sentiment_model, task3_best_model, text_model
from src.models.fusion import FusionModel

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

TEXT_PIPELINE_PATH = MODELS_DIR / "text_pipeline.joblib"
META_BUNDLE_PATH = MODELS_DIR / "meta_bundle.joblib"
PRIOR_PATH = MODELS_DIR / "prior_model.joblib"
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
    proba_prior: float
    weight_text: float
    weight_meta: float
    weight_prior: float
    model_name: str = "Fusion stacker"


class FusedPredictor:
    def __init__(self):
        self.text_pipeline = joblib.load(TEXT_PIPELINE_PATH)
        meta_bundle = joblib.load(META_BUNDLE_PATH)
        self.meta_pipeline = meta_bundle["pipeline"]
        self.top_brands = meta_bundle["top_brands"]
        self.prior: prior_model.PriorModel = joblib.load(PRIOR_PATH)
        self.fusion: FusionModel = joblib.load(FUSION_PATH)

    def predict(self, review: dict, product: dict) -> Prediction:
        """review: title, review_text, review_rating
        product: brand_name, price, avg_product_rating, product_rating_count,
                 product_tags, product_id
        """
        p_text = text_model.predict_proba_one(
            self.text_pipeline,
            review.get("review_title", ""),
            review.get("review_text", ""),
        )
        meta_row = {
            "review_rating": review.get("review_rating"),
            "review_title": review.get("review_title", ""),
            "review_text": review.get("review_text", ""),
            "price": product.get("price"),
            "avg_product_rating": product.get("avg_product_rating"),
            "product_rating_count": product.get("product_rating_count"),
            "product_tags": product.get("product_tags", ""),
            "brand_name": product.get("brand_name"),
        }
        p_meta = meta_model.predict_proba_one(self.meta_pipeline, self.top_brands, meta_row)
        p_prior = self.prior.predict_proba_one(product.get("product_id"), product.get("brand_name"))
        p_fused = self.fusion.predict_proba(p_text, p_meta, p_prior)
        return Prediction(
            proba=p_fused,
            label=p_fused >= 0.5,
            proba_text=p_text,
            proba_meta=p_meta,
            proba_prior=p_prior,
            weight_text=self.fusion.weight_text,
            weight_meta=self.fusion.weight_meta,
            weight_prior=self.fusion.weight_prior,
            model_name="Fusion stacker",
        )


class Task3BestPredictor:
    def __init__(self):
        self.model: task3_best_model.Task3BestBuyerModel = joblib.load(TASK3_BEST_PATH)

    def predict(self, review: dict, product: dict) -> Prediction:
        p = self.model.predict_proba_one(review, product)
        return Prediction(
            proba=p,
            label=p >= 0.5,
            proba_text=p,
            proba_meta=p,
            proba_prior=p,
            weight_text=1.0,
            weight_meta=0.0,
            weight_prior=0.0,
            model_name="Task 3 best: TF-IDF(1,2)+metadata LR",
        )


_PREDICTOR: Task3BestPredictor | None = None
_SENTIMENT_MODEL: sentiment_model.SentimentModel | None = None


def load_predictor() -> Task3BestPredictor:
    """Lazy singleton — safe to call from any Streamlit page."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Task3BestPredictor()
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
