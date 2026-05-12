# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

RMIT COSC3801/3015 Assignment 3, Milestone II — "GlowMate Beauty Store", an NLP-powered beauty product review and recommendation web app built with Streamlit. The repo is currently a skeleton (only `app.py`, `requirements.txt`, `README.txt`, the assignment brief PDF, and the `MAI_Group5/` Milestone I outputs exist). The runtime directories referenced in `README.txt` (`data/`, `models/`, `pages/`) have not been created yet — they need to be added as features are implemented.

The full task spec is in `AP4DS_2026A_A3 - Milestone 2_WebApp_new.pdf`. Read it before making architectural decisions.

Team: MAI_Group 5 — Pham Huynh Ngoc Hue (s3702554), Mai Thanh Nga (s4217156), Tran Thi Thao Vy (s4177924), Adam Boufous (s4215119).

## Commands

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate            # macOS/Linux
pip install -r requirements.txt

# Run the app (Streamlit auto-reloads on file change)
streamlit run app.py
```

No test framework or linter is configured. If adding tests, pick one and update this file.

## Architecture (planned, per README.txt and assignment brief)

The app is a Streamlit multi-page application. `app.py` is the landing page; additional features live as separate scripts under `pages/` (Streamlit's convention — each file becomes a sidebar entry automatically).

Four features map to four pages:
- **Task 1** — Product browsing + fuzzy keyword search across brand/name/description (uses `rapidfuzz`).
- **Task 2** — Review creation form. Predicts a recommendation label from the review text and lets the user override before saving.
- **Task 3** — Similar-item recommendations using TF-IDF cosine similarity (scikit-learn).
- **Task 4** — Admin analytics dashboard (Plotly) over review sentiment/label/rating patterns.

### HD extension: fused prediction

The recommendation label in Task 2 is intended to be produced by a **fused architecture** combining three evidence sources, each trained in Milestone I:
1. Review text model
2. Rating / metadata model
3. Product-history prior model

A `models/model_loader.py` module is expected to load these trained artifacts (joblib is in `requirements.txt` for this). When implementing, treat `model_loader.py` as the single integration point — pages should import predictors from it rather than loading pickles directly.

### Data

`data/products.csv` is the product catalog. The skeleton ships placeholder data; it must be replaced with the Milestone I dataset before submission. Reviews created via the Task 2 form should persist somewhere under `data/` (format TBD — pick CSV or SQLite and document it here when chosen).

## Milestone I artifacts (`MAI_Group5/`)

`MAI_Group5/` is the upstream Milestone I deliverable and the source of truth for the dataset, preprocessing, and the trained models the webapp must wrap. Treat it as read-only input — do not edit notebooks/scripts here as part of webapp work; instead copy or reference what you need into `data/` and `models/`.

Key files:
- `cosmetics_beauty_products_reviews.csv` — raw reviews dataset (~21 MB).
- `processed.csv` — preprocessed reviews with `review_text` replaced by cleaned token strings (output of `task1`).
- `vocab.txt` — `word:index` vocabulary used by all Milestone I feature representations.
- `count_vectors.txt`, `unweighted_vectors.txt`, `weighted_vectors.txt` — BoW counts and GloVe (unweighted / TF-IDF-weighted) document vectors. The weighted/unweighted vector files are ~175 MB each, so do NOT load them eagerly at import time; lazy-load or precompute reduced artifacts under `models/`.
- `stopwords_en.txt` — the exact stopword list used; reuse it verbatim if reproducing preprocessing in the webapp.
- `task1.py` / `task1.ipynb` — preprocessing pipeline (regex `[a-zA-Z]+(?:[-'][a-zA-Z]+)?`, lowercasing, stopword removal, singleton + top-20 DF removal, alphabetically indexed vocab). The webapp must replicate this exactly before feeding new reviews to the trained models.
- `task2_3.py` / `task2_3.ipynb` — feature construction (BoW, unweighted GloVe, TF-IDF-weighted GloVe via `gensim.downloader`) and classifiers (LogisticRegression, LinearSVC, RandomForest) with 5-fold stratified CV. The target label is "buyer vs non-buyer", which is what Task 2 of the webapp predicts.
- `MAI_Group 05.xlsx` — group submission spreadsheet (not used at runtime).

Milestone I was developed under Python 3.12 with Jupyter 7.2.2; if `requirements.txt` versions diverge from what was used to train the pickled models, expect unpickling issues.

## Pinned dependencies

`requirements.txt` pins exact versions (streamlit 1.37.1, pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.1, joblib 1.4.2, rapidfuzz 3.9.6, plotly 5.23.0). Note: Milestone I uses `gensim` (for GloVe embeddings) and `scipy.sparse` / `matplotlib`, none of which are in `requirements.txt` yet — if the webapp needs to recompute GloVe features at runtime (rather than load precomputed vectors), add `gensim` and `scipy` to the pins and document the exact GloVe model name used.
