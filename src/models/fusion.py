"""Late-fusion of (text, meta, prior) probabilities into a single label.

Two strategies; we use the LR-stacker. Weights are learned on held-out
out-of-fold predictions to avoid leakage from any single base model.
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
    weight_prior: float
    intercept: float

    def predict_proba(self, p_text: float, p_meta: float, p_prior: float) -> float:
        x = np.array([[p_text, p_meta, p_prior]])
        proba = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        pos_idx = classes.index(True) if True in classes else classes.index(1)
        return float(proba[pos_idx])


def fit(oof: np.ndarray, y) -> FusionModel:
    """oof: shape (n_samples, 3) with columns (p_text, p_meta, p_prior)."""
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(oof, y)
    coefs = clf.coef_[0]
    # Make the weights interpretable for the UI breakdown.
    s = float(np.sum(np.abs(coefs))) or 1.0
    w = np.abs(coefs) / s
    return FusionModel(
        clf=clf,
        weight_text=float(w[0]),
        weight_meta=float(w[1]),
        weight_prior=float(w[2]),
        intercept=float(clf.intercept_[0]),
    )
