"""Build dense product vectors for the Task 3 hybrid recommender.

Reads Milestone-1 outputs:

    <data_dir>/weighted_vectors.txt    per-review FastText x TF-IDF, 300-d
    <data_dir>/reviews.csv             review_id -> product_id mapping + rating

Writes:

    models/product_vectors.npy         (N, 300) L2-normalised dense vectors
    models/product_vector_index.json   ordered list of product_ids

Per-product vector = rating-weighted mean of its review vectors, with
verified buyers counting 1.5x. See notebooks/Task3_Recommendation.ipynb
for the design write-up.

Run once before launching the webapp (or skip — the recommender falls back
to a 3-signal hybrid if these artifacts are missing):

    python scripts/build_recommender.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REVIEWS_CSV     = DATA_DIR / "reviews.csv"
WEIGHTED_VEC    = DATA_DIR / "weighted_vectors.txt"
UNWEIGHTED_VEC  = DATA_DIR / "unweighted_vectors.txt"
OUT_VECTORS     = MODELS_DIR / "product_vectors.npy"
OUT_INDEX       = MODELS_DIR / "product_vector_index.json"


def log(msg: str) -> None:
    print(f"[build_recommender] {msg}", flush=True)


def stream_review_vectors(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (review_id, np.float32[D]) from the M1 vectors file.

    Each line: ``#<review_id>,v1,v2,...,vD``
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                comma = line.find(",")
                rid = line[1:comma]
                rest = line[comma + 1:]
            else:
                comma = line.find(",")
                if comma < 0:
                    continue
                rid = line[:comma]
                rest = line[comma + 1:]
            vec = np.fromstring(rest, sep=",", dtype=np.float32)
            yield rid, vec


def main() -> int:
    # Pick the weighted vectors if available (TF-IDF x FastText is the
    # canonical M1 choice); otherwise fall back to unweighted.
    if WEIGHTED_VEC.exists():
        vec_path = WEIGHTED_VEC
    elif UNWEIGHTED_VEC.exists():
        vec_path = UNWEIGHTED_VEC
    else:
        log("ERROR: no M1 vectors file found.")
        log(f"  Tried: {WEIGHTED_VEC}")
        log(f"         {UNWEIGHTED_VEC}")
        log("  (Get them from OneDrive, place in data/, and re-run.)")
        log("  The recommender will still work without these — it will run as")
        log("  a 3-signal hybrid (lexical + attribute + numeric) instead.")
        return 0

    if not REVIEWS_CSV.exists():
        log(f"ERROR: {REVIEWS_CSV} not found. Run scripts/build_catalog.py first.")
        return 2

    log(f"Reading {REVIEWS_CSV.name} for review_id -> product_id mapping ...")
    reviews = pd.read_csv(
        REVIEWS_CSV,
        dtype={"product_id": str, "review_id": str},
        keep_default_na=False,
        on_bad_lines="skip",
    )
    reviews["review_rating"] = pd.to_numeric(reviews["review_rating"], errors="coerce")

    # rating-weighted, buyer-boosted weight per review
    def _weight(rating, is_buyer) -> float:
        r = max(float(rating if rating else 3.0), 1.0) / 5.0
        return r * (1.5 if str(is_buyer).upper() == "TRUE" else 1.0)

    rev_meta: dict[str, tuple[str, float]] = {
        str(rid): (str(pid), _weight(r, b))
        for rid, pid, r, b in zip(
            reviews["review_id"],
            reviews["product_id"],
            reviews["review_rating"].fillna(3.0),
            reviews.get("is_a_buyer", pd.Series("", index=reviews.index)),
        )
    }

    product_ids = sorted(reviews["product_id"].astype(str).unique().tolist())
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # First read peeks at one line to learn the embedding dimension
    DIM = None
    for _, v in stream_review_vectors(vec_path):
        DIM = len(v)
        break
    if DIM is None:
        log("ERROR: vectors file is empty.")
        return 3
    log(f"Embedding dimension: {DIM}")

    sums    = np.zeros((len(product_ids), DIM), dtype=np.float64)
    weights = np.zeros(len(product_ids), dtype=np.float64)

    log(f"Streaming review vectors from {vec_path.name} ...")
    n_seen, n_kept = 0, 0
    t0 = time.time()
    for rid, vec in stream_review_vectors(vec_path):
        n_seen += 1
        meta = rev_meta.get(rid)
        if meta is None or len(vec) != DIM:
            continue
        pid, w = meta
        idx = pid_to_idx.get(pid)
        if idx is None:
            continue
        sums[idx]    += w * vec
        weights[idx] += w
        n_kept += 1
        if n_seen % 10000 == 0:
            log(f"  ... {n_seen:,} vectors processed ({time.time()-t0:0.1f}s)")

    weights[weights == 0] = 1.0
    prod_vecs = (sums / weights[:, None]).astype(np.float32)
    norms = np.linalg.norm(prod_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    prod_vecs /= norms

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_VECTORS, prod_vecs)
    OUT_INDEX.write_text(json.dumps(product_ids))
    log(f"Aggregated {n_kept:,} review vectors into {len(product_ids):,} products")
    log(f"Wrote {OUT_VECTORS.relative_to(ROOT)}  ({OUT_VECTORS.stat().st_size/1024:0.1f} KB)")
    log(f"Wrote {OUT_INDEX.relative_to(ROOT)}    ({OUT_INDEX.stat().st_size/1024:0.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
