"""Late-fusion of (text, meta) probabilities into a single label.

The prior channel was removed: ``prior.predict_proba_one(product_id, ...)``
returned the historical buyer rate of the *same* product, so on popular
products the fused score was dominated by product identity rather than review
content. With prior dropped and the meta model restricted to review-level
features, both base channels now read the review.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class FusionModel:
    clf: LogisticRegression
    weight_text: float
    weight_meta: float
    intercept: float
    # Ngưỡng quyết định "verified buyer" (proba > decision_threshold ⇒ True).
    # Chọn từ OOF sweep trên train set (xem scripts/train.py::select_threshold)
    # để cân bằng F1 macro và accuracy, không hardcode 0.5. Initial value khi
    # fit là train base rate; được overwrite bằng chosen threshold sau sweep.
    decision_threshold: float = 0.5

    def predict_proba(self, p_text: float, p_meta: float) -> float:
        x = np.array([[p_text, p_meta]])
        proba = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        pos_idx = classes.index(True) if True in classes else classes.index(1)
        return float(proba[pos_idx])


def fit(oof: np.ndarray, y) -> FusionModel:
    """oof: shape (n_samples, 2) with columns (p_text, p_meta)."""
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(oof, y)
    coefs = clf.coef_[0]
    s = float(np.sum(np.abs(coefs))) or 1.0
    w = np.abs(coefs) / s
    return FusionModel(
        clf=clf,
        weight_text=float(w[0]),
        weight_meta=float(w[1]),
        intercept=float(clf.intercept_[0]),
        decision_threshold=float(np.mean(np.asarray(y, dtype=bool))),
    )
