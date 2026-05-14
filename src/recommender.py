"""Task 3 — Hybrid item-item recommender for HD-level similar-items panel.

Upgrades the original ``src.similarity`` implementation (TF-IDF cosine only)
to a 4-signal fused score with Maximal-Marginal-Relevance diversity
re-ranking:

    sim(i, j) = w_sem * semantic   (FastText product embeddings, cosine)
              + w_lex * lexical    (TF-IDF product rows, cosine)
              + w_att * attribute  (brand + category match)
              + w_num * numeric    (log-price + rating proximity)

with default weights (0.50, 0.20, 0.20, 0.10). The semantic signal requires
dense product vectors built by ``scripts/build_recommender.py``; if those
are not yet built the engine **gracefully falls back** to a 3-signal hybrid
(lexical + attribute + numeric) so the webapp keeps working.

See ``notebooks/Task3_Recommendation.ipynb`` for the full design write-up:
vector representation, fusion justification, MMR maths, evaluation.

Public API
----------
``similar_products(product_id, products, k=8, use_mmr=True, mmr_lambda=0.7,
weights=None)`` returns a ``pandas.DataFrame`` with one row per
recommendation. Extra columns added on top of the input ``products`` schema:

    _score              fused similarity in [0, 1]
    _score_semantic     [0, 1]   0.0 if dense vectors not available
    _score_lexical      [0, 1]
    _score_attribute    [0, 1]
    _score_numeric      [0, 1]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.loader import load_product_tfidf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PRODUCT_VECTORS_NPY   = MODELS_DIR / "product_vectors.npy"
PRODUCT_VECTORS_INDEX = MODELS_DIR / "product_vector_index.json"


# ---------------------------------------------------------------------------
# Category inference (used by the attribute signal). The team's products.csv
# does not store a category, so we infer one from product_title + tags using
# the same patterns as the M1 preprocessing. Brand strings are stripped first
# so brand names don't trigger false category matches.
# ---------------------------------------------------------------------------
_CAT_PATTERNS = [
    ("lipstick",    r"\blipstick|\blip color|\blip stick|\blip crayon"),
    ("lip_balm",    r"\blip\s*balm|\blip butter"),
    ("foundation",  r"\bfoundation"),
    ("concealer",   r"\bconcealer"),
    ("mascara",     r"\bmascara"),
    ("eyeliner",    r"\beyeliner|\beye\s*liner|\bkajal"),
    ("eyeshadow",   r"\beyeshadow|\beye shadow|\bpalette"),
    ("blush",       r"\bblush"),
    ("highlighter", r"\bhighlighter"),
    ("compact",     r"\bcompact|\bpowder"),
    ("primer",      r"\bprimer"),
    ("nail",        r"\bnail|\bpolish"),
    ("perfume",     r"\bperfume|\beau de|\bedp|\bedt|\bbody mist|\bspray"),
    ("face_wash",   r"\bface\s*wash|\bcleanser|\bfacewash"),
    ("moisturiser", r"\bmoisturi[sz]er|\bcream|\blotion|\bhydrate"),
    ("serum",       r"\bserum|\bampoule"),
    ("sunscreen",   r"\bsunscreen|\bspf"),
    ("mask",        r"\bmask|\bsheet mask"),
    ("scrub",       r"\bscrub|\bexfoliat"),
    ("toner",       r"\btoner"),
    ("shampoo",     r"\bshampoo"),
    ("conditioner", r"\bconditioner"),
    ("hair_oil",    r"\bhair\s*oil|\bargan"),
    ("body_wash",   r"\bbody\s*wash|\bshower gel"),
    ("body_lotion", r"\bbody\s*lotion|\bbody\s*butter"),
    ("deodorant",   r"\bdeodorant|\broll[\s-]*on"),
]


def _infer_category(title: str, tags: str, brand: str) -> str:
    hay = f"{title or ''} {tags or ''}".lower()
    if brand:
        hay = hay.replace(str(brand).lower(), " ")
    for label, pat in _CAT_PATTERNS:
        if re.search(pat, hay):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Engine state (cached at module level — heavy artifacts loaded only once)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _EngineState:
    tfidf_mat: sparse.csr_matrix
    tfidf_ids: list[str]
    pid_to_idx_tfidf: dict[str, int]
    dense_mat: Optional[np.ndarray]            # (N, D), L2-normalised; None if not built
    dense_ids: Optional[list[str]]
    pid_to_idx_dense: Optional[dict[str, int]]


def _try_load_dense_vectors() -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Load the dense product vectors if they exist; otherwise return (None, None).

    Defensive normalisation: even after np.nan_to_num, a row whose norm is
    a near-zero subnormal float would, when divided into the row in float32,
    overflow to +/-Inf. So we (a) scrub all non-finite entries first,
    (b) clip tiny norms to 1.0, (c) clip any resulting non-finite entries
    in the normalised output back to 0.0. The result is guaranteed to be
    finite and unit-norm (or exactly zero, for degenerate rows).
    """
    if not PRODUCT_VECTORS_NPY.exists() or not PRODUCT_VECTORS_INDEX.exists():
        return None, None
    mat = np.load(PRODUCT_VECTORS_NPY).astype(np.float64)   # extra headroom for the norm calc
    ids = json.loads(PRODUCT_VECTORS_INDEX.read_text())

    # Step 1 — scrub NaN / +/-Inf BEFORE doing any arithmetic on it.
    mat[~np.isfinite(mat)] = 0.0

    # Step 2 — compute row norms in float64; replace anything non-finite
    # *or* numerically too small (subnormal-ish) with 1.0 to avoid an
    # explosive divide.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    bad = (~np.isfinite(norms)) | (norms < 1e-9)
    norms[bad] = 1.0

    mat = mat / norms

    # Step 3 — last-line safety net: clobber any value that *still* isn't
    # finite (e.g. from a pathological original row of all 1e308 values).
    mat[~np.isfinite(mat)] = 0.0

    return mat.astype(np.float32), list(map(str, ids))


@lru_cache(maxsize=1)
def _load_engine() -> _EngineState:
    tfidf_mat, tfidf_ids = load_product_tfidf()
    tfidf_ids = list(map(str, tfidf_ids))
    pid_to_idx_tfidf = {pid: i for i, pid in enumerate(tfidf_ids)}

    dense_mat, dense_ids = _try_load_dense_vectors()
    pid_to_idx_dense = (
        {pid: i for i, pid in enumerate(dense_ids)} if dense_ids is not None else None
    )

    return _EngineState(
        tfidf_mat=tfidf_mat.tocsr(),
        tfidf_ids=tfidf_ids,
        pid_to_idx_tfidf=pid_to_idx_tfidf,
        dense_mat=dense_mat,
        dense_ids=dense_ids,
        pid_to_idx_dense=pid_to_idx_dense,
    )


# ---------------------------------------------------------------------------
# Per-products-dataframe state (small; rebuilt cheaply on changes)
# ---------------------------------------------------------------------------
def _prepare_products(products: pd.DataFrame) -> pd.DataFrame:
    """Add a normalised brand, inferred category, and min-max numeric columns.

    Always returns a fresh DataFrame so the caller's frame is not mutated."""
    df = products.copy()
    df["_pid_str"]  = df["product_id"].astype(str)
    df["_brand_lc"] = df.get("brand_name", "").fillna("").astype(str).str.lower()

    df["_category"] = [
        _infer_category(t, tg, b)
        for t, tg, b in zip(
            df.get("product_title", pd.Series("", index=df.index)).fillna(""),
            df.get("product_tags",  pd.Series("", index=df.index)).fillna(""),
            df["_brand_lc"],
        )
    ]

    def _minmax(s: pd.Series, *, log: bool = False) -> np.ndarray:
        v = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        if log:
            v = np.log1p(np.maximum(v, 0))
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi - lo < 1e-9:
            return np.zeros_like(v, dtype=np.float32)
        return ((v - lo) / (hi - lo)).astype(np.float32)

    df["_price_norm"]  = _minmax(df.get("price", pd.Series(0, index=df.index)), log=True)
    df["_rating_norm"] = _minmax(df.get("avg_product_rating", pd.Series(0, index=df.index)))
    return df


# ---------------------------------------------------------------------------
# Similarity signals — every signal returns a vector indexed by tfidf_ids,
# i.e. the same order/length as engine.tfidf_mat rows.
# ---------------------------------------------------------------------------
def _lexical_sim(qi_tfidf: int, eng: _EngineState) -> np.ndarray:
    q = eng.tfidf_mat[qi_tfidf]
    sims = (eng.tfidf_mat @ q.T).toarray().ravel()
    sims[qi_tfidf] = 0.0
    return np.clip(sims, 0.0, 1.0).astype(np.float32)


def _semantic_sim(query_pid: str, eng: _EngineState) -> np.ndarray:
    """Dense cosine similarity, projected onto tfidf_ids ordering.

    Products that exist in tfidf_ids but NOT in dense_ids get 0.0 (the
    overall fused score still has 3 signals to work with)."""
    n = len(eng.tfidf_ids)
    if eng.dense_mat is None or eng.pid_to_idx_dense is None:
        return np.zeros(n, dtype=np.float32)
    q_idx = eng.pid_to_idx_dense.get(str(query_pid))
    if q_idx is None:
        return np.zeros(n, dtype=np.float32)
    # The matrix is sanitised at load time, but float32 BLAS routines
    # occasionally emit harmless underflow/overflow warnings from
    # vectorised micro-ops. Silence them — the result we care about is
    # the dot products against the query, which are finite by construction.
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        cos = eng.dense_mat @ eng.dense_mat[q_idx]            # (N_dense,)
    cos = np.nan_to_num(cos, nan=0.0, posinf=1.0, neginf=-1.0)
    sims = np.zeros(n, dtype=np.float32)
    for i, pid in enumerate(eng.tfidf_ids):
        j = eng.pid_to_idx_dense.get(pid)
        if j is not None:
            sims[i] = (cos[j] + 1.0) / 2.0                     # (-1,1) -> (0,1)
    qi_tfidf = eng.pid_to_idx_tfidf.get(str(query_pid))
    if qi_tfidf is not None:
        sims[qi_tfidf] = 0.0
    return np.clip(sims, 0.0, 1.0)


def _attribute_sim(query_pid: str, prep: pd.DataFrame, eng: _EngineState) -> np.ndarray:
    """Brand + category match, projected to tfidf_ids order."""
    pid_to_row = {pid: i for i, pid in enumerate(prep["_pid_str"].tolist())}
    q_idx = pid_to_row.get(str(query_pid))
    n = len(eng.tfidf_ids)
    if q_idx is None:
        return np.zeros(n, dtype=np.float32)
    q_brand = prep["_brand_lc"].iloc[q_idx]
    q_cat   = prep["_category"].iloc[q_idx]

    brand_arr = prep["_brand_lc"].to_numpy()
    cat_arr   = prep["_category"].to_numpy()
    same_brand = (brand_arr == q_brand).astype(np.float32)
    same_cat   = (cat_arr   == q_cat).astype(np.float32)
    per_product_score = 0.5 * same_brand + 0.5 * same_cat

    sims = np.zeros(n, dtype=np.float32)
    for i, pid in enumerate(eng.tfidf_ids):
        r = pid_to_row.get(pid)
        if r is not None:
            sims[i] = per_product_score[r]
    qi_tfidf = eng.pid_to_idx_tfidf.get(str(query_pid))
    if qi_tfidf is not None:
        sims[qi_tfidf] = 0.0
    return sims


def _numeric_sim(query_pid: str, prep: pd.DataFrame, eng: _EngineState) -> np.ndarray:
    pid_to_row = {pid: i for i, pid in enumerate(prep["_pid_str"].tolist())}
    q_idx = pid_to_row.get(str(query_pid))
    n = len(eng.tfidf_ids)
    if q_idx is None:
        return np.zeros(n, dtype=np.float32)
    qp = prep["_price_norm"].iloc[q_idx]
    qr = prep["_rating_norm"].iloc[q_idx]

    p = prep["_price_norm"].to_numpy()
    r = prep["_rating_norm"].to_numpy()
    num = 1.0 - 0.6 * np.abs(p - qp) - 0.4 * np.abs(r - qr)
    num = np.clip(num, 0.0, 1.0).astype(np.float32)

    sims = np.zeros(n, dtype=np.float32)
    for i, pid in enumerate(eng.tfidf_ids):
        rr = pid_to_row.get(pid)
        if rr is not None:
            sims[i] = num[rr]
    qi_tfidf = eng.pid_to_idx_tfidf.get(str(query_pid))
    if qi_tfidf is not None:
        sims[qi_tfidf] = 0.0
    return sims


# ---------------------------------------------------------------------------
# Default weights — re-normalised when the semantic signal is unavailable.
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS = {
    "semantic":  0.50,
    "lexical":   0.20,
    "attribute": 0.20,
    "numeric":   0.10,
}
_FALLBACK_WEIGHTS = {                # used when dense vectors are not built
    "semantic":  0.00,
    "lexical":   0.50,
    "attribute": 0.30,
    "numeric":   0.20,
}


def _resolve_weights(weights: Optional[dict], has_dense: bool) -> dict:
    if weights:
        s = sum(weights.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {s:.4f}")
        return dict(weights)
    return dict(_DEFAULT_WEIGHTS) if has_dense else dict(_FALLBACK_WEIGHTS)


# ---------------------------------------------------------------------------
# Maximal Marginal Relevance re-ranking (uses the lexical signal as the
# pairwise diversity surface when dense vectors aren't built).
# ---------------------------------------------------------------------------
def _mmr_rerank(
    candidate_idxs: list[int],
    fused: np.ndarray,
    pair_sim_fn,
    k: int,
    lam: float,
) -> list[int]:
    selected: list[int] = []
    remaining = list(candidate_idxs)
    while remaining and len(selected) < k:
        best, best_score = None, -1e18
        for idx in remaining:
            rel = fused[idx]
            div = 0.0 if not selected else max(pair_sim_fn(idx, s) for s in selected)
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best, best_score = idx, score
        selected.append(best)
        remaining.remove(best)
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def similar_products(
    product_id,
    products: pd.DataFrame,
    k: int = 8,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    weights: Optional[dict] = None,
    n_candidates: int = 80,
) -> pd.DataFrame:
    """Return up to ``k`` recommendations for ``product_id`` from ``products``.

    Parameters
    ----------
    product_id : Any
        The id of the anchor product (coerced to ``str``).
    products : DataFrame
        The catalogue. Must contain at least ``product_id``, ``brand_name``,
        ``product_title``, ``product_tags``, ``price``, ``avg_product_rating``.
    k : int
        How many recommendations to return.
    use_mmr : bool
        If True, top ``n_candidates`` by fused relevance are re-ranked with
        MMR (lambda=``mmr_lambda``) for diversity.
    weights : dict[str, float] | None
        Optional override of the {semantic, lexical, attribute, numeric}
        weights; must sum to 1.0.
    """
    eng = _load_engine()
    pid_str = str(product_id)
    if pid_str not in eng.pid_to_idx_tfidf:
        return products.iloc[0:0].assign(
            _score=[], _score_semantic=[], _score_lexical=[],
            _score_attribute=[], _score_numeric=[],
        )

    prep = _prepare_products(products)
    qi_tfidf = eng.pid_to_idx_tfidf[pid_str]

    sem = _semantic_sim(pid_str, eng)
    lex = _lexical_sim(qi_tfidf, eng)
    attr = _attribute_sim(pid_str, prep, eng)
    num  = _numeric_sim(pid_str, prep, eng)

    w = _resolve_weights(weights, has_dense=eng.dense_mat is not None)
    fused = (
        w["semantic"]  * sem
        + w["lexical"]   * lex
        + w["attribute"] * attr
        + w["numeric"]   * num
    ).astype(np.float32)

    # Top-N candidates by fused score, then MMR if requested
    n = len(fused)
    n_cand = max(k, min(n_candidates, n - 1))
    cand = np.argpartition(-fused, min(n_cand, n - 1))[:n_cand]
    cand = cand[np.argsort(-fused[cand])].tolist()

    if use_mmr:
        # Diversity penalty surface for MMR. We combine TWO signals:
        #
        #   - dense semantic similarity (smooth, captures "review vibe")
        #   - attribute similarity      (hard 0/0.5/1.0 on brand+category)
        #
        # and take the MAX. Either being high counts as "too similar" for
        # diversification. This is what lets MMR actively spread the list
        # across brands and categories at low lambda — without this, the
        # embeddings cluster too tightly within a category and lowering
        # lambda does not visibly change the ranking.
        pid_to_prep_row = {pid: i for i, pid in enumerate(prep["_pid_str"].tolist())}
        brand_arr = prep["_brand_lc"].to_numpy()
        cat_arr   = prep["_category"].to_numpy()

        def _attr_pair(i: int, j: int) -> float:
            pid_i, pid_j = eng.tfidf_ids[i], eng.tfidf_ids[j]
            ri = pid_to_prep_row.get(pid_i)
            rj = pid_to_prep_row.get(pid_j)
            if ri is None or rj is None:
                return 0.0
            return (
                0.5 * float(brand_arr[ri] == brand_arr[rj])
                + 0.5 * float(cat_arr[ri]   == cat_arr[rj])
            )

        if eng.dense_mat is not None:
            tfidf_to_dense = [
                eng.pid_to_idx_dense.get(pid) for pid in eng.tfidf_ids
            ]

            def pair_sim(i: int, j: int) -> float:
                di, dj = tfidf_to_dense[i], tfidf_to_dense[j]
                if di is None or dj is None:
                    sem = float((eng.tfidf_mat[i] @ eng.tfidf_mat[j].T).toarray().ravel()[0])
                else:
                    sem = float((eng.dense_mat[di] @ eng.dense_mat[dj] + 1.0) / 2.0)
                return max(sem, _attr_pair(i, j))
        else:
            def pair_sim(i: int, j: int) -> float:
                lex = float((eng.tfidf_mat[i] @ eng.tfidf_mat[j].T).toarray().ravel()[0])
                return max(lex, _attr_pair(i, j))

        chosen = _mmr_rerank(cand, fused, pair_sim, k=k, lam=float(mmr_lambda))
    else:
        chosen = cand[:k]

    # Materialise the output DataFrame using the input products schema
    rec_pids = [eng.tfidf_ids[i] for i in chosen]
    prep_indexed = prep.set_index("_pid_str")
    out = prep_indexed.loc[[p for p in rec_pids if p in prep_indexed.index]].reset_index(drop=True)
    if len(out) == 0:
        return out

    # Attach scores in the same order
    score_lookup = {eng.tfidf_ids[i]: i for i in chosen}
    out["_score"]           = [float(fused[score_lookup[pid]])  for pid in rec_pids if pid in prep_indexed.index]
    out["_score_semantic"]  = [float(sem  [score_lookup[pid]])  for pid in rec_pids if pid in prep_indexed.index]
    out["_score_lexical"]   = [float(lex  [score_lookup[pid]])  for pid in rec_pids if pid in prep_indexed.index]
    out["_score_attribute"] = [float(attr [score_lookup[pid]])  for pid in rec_pids if pid in prep_indexed.index]
    out["_score_numeric"]   = [float(num  [score_lookup[pid]])  for pid in rec_pids if pid in prep_indexed.index]

    # Don't leak the internal prep columns to the caller
    return out.drop(columns=[c for c in ("_brand_lc", "_category", "_price_norm", "_rating_norm") if c in out.columns])


def has_dense_vectors() -> bool:
    """UI helper — lets the page show a small warning when M1 vectors haven't been built."""
    return _load_engine().dense_mat is not None
