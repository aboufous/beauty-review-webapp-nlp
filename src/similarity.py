"""Task 3 item-to-item similarity via TF-IDF cosine."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.loader import load_product_tfidf


def similar_products(product_id, products: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    mat, ids = load_product_tfidf()
    pid_str = str(product_id)
    if pid_str not in ids:
        return products.iloc[0:0].assign(_score=[])

    row_idx = ids.index(pid_str)
    sims = cosine_similarity(mat[row_idx], mat).ravel()
    order = sims.argsort()[::-1]
    out_ids = []
    out_scores = []
    for i in order:
        if ids[i] == pid_str:
            continue
        out_ids.append(ids[i])
        out_scores.append(float(sims[i]))
        if len(out_ids) >= k:
            break

    products = products.copy()
    products["_pid_str"] = products["product_id"].astype(str)
    out = products.set_index("_pid_str").loc[out_ids].reset_index(drop=True)
    out["_score"] = out_scores
    return out
