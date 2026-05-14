"""Task 3 item-to-item similarity.

The original TF-IDF-only implementation has been upgraded to a 4-signal
hybrid recommender. The new implementation lives in ``src.recommender``;
this module is a backward-compatible shim so any existing imports of
``from src.similarity import similar_products`` keep working.

See ``notebooks/Task3_Recommendation.ipynb`` for the full design rationale.
"""
from __future__ import annotations

from src.recommender import similar_products, has_dense_vectors  # noqa: F401

__all__ = ["similar_products", "has_dense_vectors"]
