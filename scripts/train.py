"""Train text + meta + prior models, fit fusion, build Task-3 TF-IDF matrix.

Run once after `scripts/build_catalog.py`:

    python scripts/train.py

Writes everything into ``models/``. Pages load via ``src.models.loader``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import REVIEWS_CSV, PRODUCTS_CSV  # noqa: E402
from src.models import fusion, meta_model, prior_model, text_model  # noqa: E402
from src.models.loader import (  # noqa: E402
    FUSION_PATH,
    META_BUNDLE_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PRIOR_PATH,
    PRODUCT_INDEX_PATH,
    PRODUCT_TFIDF_PATH,
    TEXT_PIPELINE_PATH,
    TFIDF_PATH,
)
from src.preprocess import clean_text  # noqa: E402

SEED = 42


def _macro_f1(y, p):
    return f1_score(y, p >= 0.5, average="macro")


def _proba_pos(estimator, X):
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    pos_idx = classes.index(True) if True in classes else classes.index(1)
    return proba[:, pos_idx]


def train_text(df_train, y_train, df_test, y_test):
    print("\n[text]  fitting TF-IDF + LogisticRegression …")
    t0 = time.time()
    pipeline = text_model.build_pipeline()
    X_train = text_model.review_corpus(df_train)
    pipeline.fit(X_train, y_train)

    # OOF for the fusion stacker — note: oof on training rows only.
    print("[text]  computing 5-fold OOF probabilities for fusion …")
    fresh = text_model.build_pipeline()
    oof = cross_val_predict(
        fresh,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        method="predict_proba",
        n_jobs=-1,
    )
    classes = list(fresh.classes_) if hasattr(fresh, "classes_") else None
    # cross_val_predict's `method='predict_proba'` returns columns in the order
    # of the unique sorted classes — for boolean it's [False, True].
    p_train_text = oof[:, -1]  # last column = True / 1

    X_test = text_model.review_corpus(df_test)
    p_test = _proba_pos(pipeline, X_test)
    print(
        f"[text]  done in {time.time()-t0:.1f}s — "
        f"test macro-F1 = {_macro_f1(y_test, p_test):.4f}, "
        f"acc = {accuracy_score(y_test, p_test >= 0.5):.4f}"
    )
    return pipeline, p_train_text, p_test


def train_meta(df_train, y_train, df_test, y_test):
    print("\n[meta]  fitting HistGradientBoosting (calibrated) …")
    t0 = time.time()
    X_train, top_brands = meta_model.derive_features(df_train)
    pipeline = meta_model.build_pipeline()
    pipeline.fit(X_train, y_train)

    print("[meta]  computing 5-fold OOF probabilities …")
    fresh = meta_model.build_pipeline()
    oof = cross_val_predict(
        fresh,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        method="predict_proba",
        n_jobs=-1,
    )
    p_train_meta = oof[:, -1]

    X_test, _ = meta_model.derive_features(df_test, top_brands=top_brands)
    p_test = _proba_pos(pipeline, X_test)
    print(
        f"[meta]  done in {time.time()-t0:.1f}s — "
        f"test macro-F1 = {_macro_f1(y_test, p_test):.4f}, "
        f"acc = {accuracy_score(y_test, p_test >= 0.5):.4f}"
    )
    return pipeline, top_brands, p_train_meta, p_test


def train_prior(df_train, y_train, df_test, y_test):
    print("\n[prior] computing smoothed product/brand priors …")
    t0 = time.time()
    train_for_prior = df_train.copy()
    train_for_prior["is_a_buyer"] = y_train.astype(bool)
    pm = prior_model.fit(train_for_prior)

    p_train = np.fromiter(
        (pm.predict_proba_one(pid, bn) for pid, bn in zip(df_train["product_id"], df_train["brand_name"])),
        dtype=float,
        count=len(df_train),
    )
    p_test = np.fromiter(
        (pm.predict_proba_one(pid, bn) for pid, bn in zip(df_test["product_id"], df_test["brand_name"])),
        dtype=float,
        count=len(df_test),
    )
    print(
        f"[prior] done in {time.time()-t0:.1f}s — "
        f"test macro-F1 = {_macro_f1(y_test, p_test):.4f}, "
        f"acc = {accuracy_score(y_test, p_test >= 0.5):.4f}"
    )
    return pm, p_train, p_test


def fit_fusion(p_train_text, p_train_meta, p_train_prior, y_train,
               p_test_text, p_test_meta, p_test_prior, y_test):
    print("\n[fusion] fitting LR stacker on OOF probabilities …")
    oof = np.column_stack([p_train_text, p_train_meta, p_train_prior])
    fm = fusion.fit(oof, y_train)

    test_stack = np.column_stack([p_test_text, p_test_meta, p_test_prior])
    p_test = _proba_pos(fm.clf, test_stack)
    print(
        f"[fusion] weights — text={fm.weight_text:.2f} meta={fm.weight_meta:.2f} prior={fm.weight_prior:.2f}"
    )
    print(
        f"[fusion] test macro-F1 = {_macro_f1(y_test, p_test):.4f}, "
        f"acc = {accuracy_score(y_test, p_test >= 0.5):.4f}"
    )
    return fm, p_test


def build_product_tfidf(text_pipeline, reviews: pd.DataFrame, products: pd.DataFrame):
    print("\n[task3] building per-product TF-IDF matrix for similarity …")
    tfidf = text_pipeline.named_steps["tfidf"]
    # Concatenate cleaned text per product (cap to bound memory).
    by_prod = (
        reviews.assign(
            _clean=(reviews["review_title"].fillna("") + " " + reviews["review_text"].fillna("")).map(clean_text)
        )
        .groupby("product_id")["_clean"]
        .apply(lambda s: " ".join(s.tolist())[:4000])
        .to_dict()
    )
    docs, ids = [], []
    for pid in products["product_id"].tolist():
        text_field = " ".join(
            str(products.loc[products["product_id"] == pid, c].iloc[0] or "")
            for c in ("brand_name", "product_title", "product_tags")
        )
        agg = by_prod.get(pid, "")
        docs.append(f"{clean_text(text_field)} {agg}")
        ids.append(pid)
    mat = tfidf.transform(docs)
    sparse.save_npz(PRODUCT_TFIDF_PATH, mat)
    PRODUCT_INDEX_PATH.write_text(json.dumps([str(x) for x in ids]))
    joblib.dump(tfidf, TFIDF_PATH)
    print(f"[task3] wrote {PRODUCT_TFIDF_PATH.name} — shape {mat.shape}")


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data …")
    reviews = pd.read_csv(REVIEWS_CSV)
    products = pd.read_csv(PRODUCTS_CSV)
    reviews = reviews.dropna(subset=["is_a_buyer", "review_text"]).reset_index(drop=True)
    reviews["is_a_buyer"] = reviews["is_a_buyer"].astype(bool)
    y = reviews["is_a_buyer"]
    df_train, df_test, y_train, y_test = train_test_split(
        reviews, y, test_size=0.15, stratify=y, random_state=SEED
    )

    text_pipeline, p_train_text, p_test_text = train_text(df_train, y_train, df_test, y_test)
    meta_pipeline, top_brands, p_train_meta, p_test_meta = train_meta(df_train, y_train, df_test, y_test)
    pm, p_train_prior, p_test_prior = train_prior(df_train, y_train, df_test, y_test)
    fm, p_test_fused = fit_fusion(
        p_train_text, p_train_meta, p_train_prior, y_train,
        p_test_text, p_test_meta, p_test_prior, y_test,
    )

    joblib.dump(text_pipeline, TEXT_PIPELINE_PATH)
    joblib.dump({"pipeline": meta_pipeline, "top_brands": top_brands}, META_BUNDLE_PATH)
    joblib.dump(pm, PRIOR_PATH)
    joblib.dump(fm, FUSION_PATH)

    build_product_tfidf(text_pipeline, reviews, products)

    metrics = {
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "test_macro_f1": {
            "text": float(_macro_f1(y_test, p_test_text)),
            "meta": float(_macro_f1(y_test, p_test_meta)),
            "prior": float(_macro_f1(y_test, p_test_prior)),
            "fused": float(_macro_f1(y_test, p_test_fused)),
        },
        "test_accuracy": {
            "text": float(accuracy_score(y_test, p_test_text >= 0.5)),
            "meta": float(accuracy_score(y_test, p_test_meta >= 0.5)),
            "prior": float(accuracy_score(y_test, p_test_prior >= 0.5)),
            "fused": float(accuracy_score(y_test, p_test_fused >= 0.5)),
        },
        "fusion_weights": {
            "text": fm.weight_text,
            "meta": fm.weight_meta,
            "prior": fm.weight_prior,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nAll artifacts under {MODELS_DIR}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
