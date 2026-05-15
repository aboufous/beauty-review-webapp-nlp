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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import PRODUCTS_CSV, REVIEWS_CSV  # noqa: E402
from src.models import (fusion, meta_model, sentiment_model,  # noqa: E402
                        task3_best_model, text_model)
from src.models.loader import (FUSION_PATH, META_BUNDLE_PATH,  # noqa: E402
                               METRICS_PATH, MODELS_DIR,
                               PRODUCT_INDEX_PATH, PRODUCT_TFIDF_PATH,
                               SENTIMENT_PATH, TASK3_BEST_PATH,
                               TEXT_PIPELINE_PATH, TFIDF_PATH)
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


def fit_fusion(p_train_text, p_train_meta, y_train,
               p_test_text, p_test_meta, y_test):
    print("\n[fusion] fitting LR stacker on OOF probabilities (text + meta) …")
    oof = np.column_stack([p_train_text, p_train_meta])
    fm = fusion.fit(oof, y_train)

    # OOF fused probability — used for threshold selection (no test-set leakage).
    p_oof_fused = _proba_pos(fm.clf, oof)

    test_stack = np.column_stack([p_test_text, p_test_meta])
    p_test = _proba_pos(fm.clf, test_stack)
    print(
        f"[fusion] weights — text={fm.weight_text:.2f} meta={fm.weight_meta:.2f}"
    )
    return fm, p_oof_fused, p_test


def select_threshold(p_oof: np.ndarray, y_oof, criterion: str = "geom_mean"):
    """Sweep candidate thresholds on OOF predictions and return the best one.

    Reports several criteria so the choice is auditable:
      - max_f1            : argmax macro-F1
      - max_accuracy      : argmax accuracy (often degenerates to majority class)
      - max_balanced_acc  : argmax (TPR+TNR)/2 (= Youden's J + 0.5)
      - max_mcc           : argmax Matthews correlation coefficient
      - geom_mean         : argmax sqrt(macro-F1 * accuracy)  (default — balances both)
      - harmonic_mean     : argmax 2·F1·acc / (F1+acc)       (penalises weaker metric)
      - base_rate         : threshold = P(y=1) on train (no sweep)

    Returns (chosen_threshold, dict_of_all_results).
    """
    y = np.asarray(y_oof, dtype=bool)
    base_rate = float(y.mean())
    grid = np.linspace(0.05, 0.95, 91)

    rows = []
    for t in grid:
        yhat = p_oof > t
        f1 = f1_score(y, yhat, average="macro", zero_division=0)
        acc = accuracy_score(y, yhat)
        bal = balanced_accuracy_score(y, yhat)
        mcc = matthews_corrcoef(y, yhat) if yhat.sum() not in (0, len(yhat)) else 0.0
        gmean = float(np.sqrt(max(f1, 0) * max(acc, 0)))
        hmean = (2 * f1 * acc / (f1 + acc)) if (f1 + acc) > 0 else 0.0
        rows.append((float(t), f1, acc, bal, mcc, gmean, hmean))

    arr = np.array(rows)  # cols: t, f1, acc, bal, mcc, gmean, hmean
    summary = {
        "max_f1":           {"threshold": float(arr[np.argmax(arr[:, 1]), 0])},
        "max_accuracy":     {"threshold": float(arr[np.argmax(arr[:, 2]), 0])},
        "max_balanced_acc": {"threshold": float(arr[np.argmax(arr[:, 3]), 0])},
        "max_mcc":          {"threshold": float(arr[np.argmax(arr[:, 4]), 0])},
        "geom_mean":        {"threshold": float(arr[np.argmax(arr[:, 5]), 0])},
        "harmonic_mean":    {"threshold": float(arr[np.argmax(arr[:, 6]), 0])},
        "base_rate":        {"threshold": base_rate},
    }
    # Attach the achieved (F1, accuracy) at each chosen threshold for transparency.
    for name, payload in summary.items():
        t = payload["threshold"]
        yhat = p_oof > t
        payload["macro_f1"] = float(f1_score(y, yhat, average="macro", zero_division=0))
        payload["accuracy"] = float(accuracy_score(y, yhat))
        payload["balanced_acc"] = float(balanced_accuracy_score(y, yhat))
        payload["mcc"] = float(matthews_corrcoef(y, yhat)) if yhat.sum() not in (0, len(yhat)) else 0.0

    chosen = float(summary[criterion]["threshold"])
    print("\n[threshold] OOF sweep — comparison of selection criteria:")
    print(f"  {'criterion':<18} {'thr':>6}  {'F1':>6}  {'acc':>6}  {'bal-acc':>7}  {'MCC':>6}")
    for name, p in summary.items():
        mark = " ←" if name == criterion else ""
        print(f"  {name:<18} {p['threshold']:>6.3f}  {p['macro_f1']:>6.3f}  "
              f"{p['accuracy']:>6.3f}  {p['balanced_acc']:>7.3f}  {p['mcc']:>6.3f}{mark}")
    print(f"[threshold] chosen criterion = {criterion!r} → threshold = {chosen:.4f}")
    return chosen, summary


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


def train_sentiment(reviews: pd.DataFrame):
    print("\n[sentiment] fitting customer sentiment model from text + rating …")
    sm = sentiment_model.train(reviews)
    joblib.dump(sm, SENTIMENT_PATH)
    print(f"[sentiment] wrote {SENTIMENT_PATH.name}")
    return sm


def train_task3_best(reviews: pd.DataFrame):
    print("\n[task3-best] fitting TF-IDF(1,2)+metadata LogisticRegression …")
    model = task3_best_model.train(reviews)
    joblib.dump(model, TASK3_BEST_PATH)
    print(f"[task3-best] wrote {TASK3_BEST_PATH.name}")
    return model


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
    fm, p_oof_fused, p_test_fused = fit_fusion(
        p_train_text, p_train_meta, y_train,
        p_test_text, p_test_meta, y_test,
    )

    # Pick a decision threshold from OOF predictions (no test-set leakage).
    # Default: geometric mean of macro-F1 and accuracy — balances both.
    chosen_thr, thr_summary = select_threshold(p_oof_fused, y_train, criterion="geom_mean")
    fm.decision_threshold = chosen_thr

    joblib.dump(text_pipeline, TEXT_PIPELINE_PATH)
    joblib.dump({"pipeline": meta_pipeline, "top_brands": top_brands}, META_BUNDLE_PATH)
    joblib.dump(fm, FUSION_PATH)
    train_task3_best(reviews)
    train_sentiment(reviews)

    build_product_tfidf(text_pipeline, reviews, products)

    thr = fm.decision_threshold
    metrics = {
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "decision_threshold": float(thr),
        "threshold_selection": thr_summary,
        "test_macro_f1": {
            "text": float(f1_score(y_test, p_test_text > thr, average="macro")),
            "meta": float(f1_score(y_test, p_test_meta > thr, average="macro")),
            "fused": float(f1_score(y_test, p_test_fused > thr, average="macro")),
        },
        "test_accuracy": {
            "text": float(accuracy_score(y_test, p_test_text > thr)),
            "meta": float(accuracy_score(y_test, p_test_meta > thr)),
            "fused": float(accuracy_score(y_test, p_test_fused > thr)),
        },
        "fusion_weights": {
            "text": fm.weight_text,
            "meta": fm.weight_meta,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nAll artifacts under {MODELS_DIR}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
