"""Task 3 best buyer classifier from the M1 notebook.

Notebook result:
TF-IDF(1,2) + one-hot product metadata + LogisticRegression
with GridSearchCV best params ``C=1.0, class_weight='balanced'``.
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
class Task3BestBuyerModel:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression
    brand_columns: list[str]
    price_median: float
    avg_rating_median: float

    def _metadata_frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        price = pd.to_numeric(df.get("price"), errors="coerce").fillna(self.price_median)
        avg_rating = pd.to_numeric(df.get("avg_product_rating"), errors="coerce").fillna(
            self.avg_rating_median
        )
        rating_count = pd.to_numeric(df.get("product_rating_count"), errors="coerce").fillna(0)
        brand = pd.get_dummies(
            df.get("brand_name", pd.Series(["Unknown"] * len(df))).fillna("Unknown"),
            prefix="brand",
            dtype=np.float32,
        )
        brand = brand.reindex(columns=self.brand_columns, fill_value=0.0)
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "price_log1p": np.log1p(price).astype(np.float32),
                        "avg_product_rating": avg_rating.astype(np.float32),
                        "rating_count_log1p": np.log1p(rating_count).astype(np.float32),
                    }
                ),
                brand.reset_index(drop=True),
            ],
            axis=1,
        )

    def _features(self, rows: list[dict]) -> sparse.csr_matrix:
        texts = [
            clean_text(f"{row.get('review_text', '') or ''} {row.get('review_title', '') or ''}")
            for row in rows
        ]
        x_text = self.vectorizer.transform(texts)
        x_extra = sparse.csr_matrix(self._metadata_frame(rows).to_numpy(dtype=np.float32))
        return sparse.hstack([x_text, x_extra], format="csr")

    def predict_proba_one(self, review: dict, product: dict) -> float:
        row = {
            "review_title": review.get("review_title", ""),
            "review_text": review.get("review_text", ""),
            "price": product.get("price"),
            "avg_product_rating": product.get("avg_product_rating"),
            "product_rating_count": product.get("product_rating_count"),
            "brand_name": product.get("brand_name"),
        }
        proba = self.clf.predict_proba(self._features([row]))[0]
        classes = list(self.clf.classes_)
        pos_idx = classes.index(True) if True in classes else classes.index(1)
        return float(proba[pos_idx])


def train(reviews: pd.DataFrame) -> Task3BestBuyerModel:
    df = reviews.dropna(subset=["is_a_buyer", "review_text"]).reset_index(drop=True).copy()
    df["is_a_buyer"] = df["is_a_buyer"].astype(bool)

    combined_text = (
        df["review_text"].fillna("").astype(str)
        + " "
        + df["review_title"].fillna("").astype(str)
    ).map(clean_text)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    x_text = vectorizer.fit_transform(combined_text.tolist())

    price_num = pd.to_numeric(df["price"], errors="coerce")
    price_median = float(price_num.median())
    avg_rating_num = pd.to_numeric(df["avg_product_rating"], errors="coerce")
    avg_rating_median = float(avg_rating_num.median())
    rating_count_num = pd.to_numeric(df["product_rating_count"], errors="coerce")

    brand_ohe = pd.get_dummies(df["brand_name"].fillna("Unknown"), prefix="brand", dtype=np.float32)
    extra_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "price_log1p": np.log1p(price_num.fillna(price_median)).astype(np.float32),
                    "avg_product_rating": avg_rating_num.fillna(avg_rating_median).astype(np.float32),
                    "rating_count_log1p": np.log1p(rating_count_num.fillna(0)).astype(np.float32),
                }
            ),
            brand_ohe.reset_index(drop=True),
        ],
        axis=1,
    )
    x = sparse.hstack([x_text, sparse.csr_matrix(extra_df.to_numpy(dtype=np.float32))], format="csr")
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(x, df["is_a_buyer"].to_numpy())
    return Task3BestBuyerModel(
        vectorizer=vectorizer,
        clf=clf,
        brand_columns=brand_ohe.columns.tolist(),
        price_median=price_median,
        avg_rating_median=avg_rating_median,
    )
