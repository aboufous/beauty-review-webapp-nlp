COSC3801/3015 Advanced Programming for Data Science
Assignment 3 — Milestone II: NLP Web-based Data Application
Group: MAI_Group 5

================================================================================
Team members
================================================================================
- Pham Huynh Ngoc Hue  - s3702554
- Mai Thanh Nga        - s4217156
- Tran Thi Thao Vy     - s4177924
- Adam Boufous         - s4215119

================================================================================
What this is
================================================================================
"GlowMate Beauty Store" — a Streamlit web application that lets shoppers
browse a cosmetics catalogue, write reviews, and see similar-item
recommendations. The recommendation-label predictor for new reviews is a
late-fusion of three independently-trained models, satisfying the
DI/HD-level requirement of combining "at least two/three different models,
which use different types of data".

Four tasks, four pages:
- Task 1  pages/1_🛍_Browse_and_Search.py   Fuzzy keyword search (rapidfuzz)
- Task 2  pages/2_📦_Product_Detail.py      Review form + fused-model label
- Task 3  pages/2_📦_Product_Detail.py      Similar items (TF-IDF cosine)
- Task 4  pages/3_📊_Admin_Dashboard.py     Plotly analytics

Search note (Task 1): the brief mentions matching against "brand name
or description". The Milestone-I dataset has no free-text description
column, so the search field concatenates brand_name + product_title +
product_tags as the closest faithful equivalent — tags carry the
category/ingredient keywords that a description would normally cover.

================================================================================
How to run
================================================================================
1. Create a virtual environment and install dependencies:

       python3 -m venv .venv
       source .venv/bin/activate           # macOS/Linux
       .venv\\Scripts\\activate            # Windows
       pip install -r requirements.txt

2. Build the data catalogue (one-off; reads MAI_Group5/):

       python scripts/build_catalog.py

   Produces data/products.csv and data/reviews.csv.

3. Train the three sub-models + the fusion stacker + the Task-3
   TF-IDF matrix (one-off; ~30s on a laptop):

       python scripts/train.py

   Writes models/*.joblib, models/product_tfidf_matrix.npz,
   models/metrics.json.

4. Launch the app:

       streamlit run app.py

   Open the URL Streamlit prints (typically http://localhost:8501).

================================================================================
Fused architecture (Task 2, HD)
================================================================================
Three independently-trained classifiers, each on a different data type:

  1. Text model    TF-IDF(1,2) + LogisticRegression(C=2.0, balanced) over
                   cleaned title+body tokens. Same M1-style preprocessing
                   pipeline (regex / lowercase / stopwords / min_df=5).
  2. Meta model    HistGradientBoosting (isotonic-calibrated) on numeric
                   features (review_rating, price, avg_product_rating,
                   product_rating_count, text/title length, n_tags) plus
                   a one-hot top-25 brand bucket. Strictly tabular — no
                   review content.
  3. Prior model   Bayesian-smoothed per-product P(is_a_buyer); falls
                   back to per-brand rate, then the global rate.

Their probabilities are stacked by a LogisticRegression meta-learner
fit on 5-fold out-of-fold predictions so the meta-learner never sees
in-sample base predictions.

Held-out test (15% stratified split, n=9,192) — see models/metrics.json:
    text   Macro-F1 0.627   acc 0.706
    meta   Macro-F1 0.715   acc 0.834
    prior  Macro-F1 0.676   acc 0.818
    fused  Macro-F1 0.741   acc 0.841

Fusion lifts Macro-F1 by ~+2.6 over the strongest base (meta). The
stacker's relative weights are text 20% · meta 59% · prior 21% — the
tabular signal carries most of the load on this dataset, with text and
prior contributing complementary lift via the stacker.

================================================================================
File layout
================================================================================
app.py                        Landing page
pages/                        Streamlit multi-page UIs
src/                          Pure-Python modules (no Streamlit imports)
  preprocess.py               Mirrors MAI_Group5/task1.py tokenisation
  data.py                     Catalogue + review I/O
  search.py                   Task 1 fuzzy search
  similarity.py               Task 3 similar-item lookup
  ui.py                       Streamlit helpers (cards, caching)
  models/                     text_model / meta_model / prior_model /
                              fusion / loader (the single integration
                              point for joblib artifacts)
scripts/
  build_catalog.py            Materialises data/products.csv + data/reviews.csv
  train.py                    Trains everything; writes models/*
data/
  products.csv                Generated; one row per product_id
  reviews.csv                 Generated; appended at runtime by Task 2
  stopwords_en.txt            Copy of MAI_Group5/stopwords_en.txt
models/                       Generated by scripts/train.py
MAI_Group5/                   READ-ONLY Milestone-I deliverable

================================================================================
Notes
================================================================================
- Product images are SVG cards generated inline — gradient colour and
  emoji chosen from the category extracted from the product title
  (lipstick → 💄, mascara → 🖤, shampoo → 🧴, …). The brief explicitly
  allows artificial display images for the purpose of showing the
  catalogue, so the synthetic SVG is the intended visual.
- New reviews are persisted to data/reviews.csv (atomic write). Pages that
  read reviews use a short-TTL Streamlit cache so saves are visible quickly.
- The fused predictor is loaded lazily as a process-level singleton via
  src/models/loader.py — pages never load joblib files directly.

================================================================================
Video demo
================================================================================
A ≤ 4-minute walk-through covering: browsing + fuzzy search, opening a
product, writing a review with override, similar items, and the admin
dashboard. File: MAI_Group5_demo.mp4 (included in the submission zip).

================================================================================
Large files (OneDrive)
================================================================================
The zipped submission exceeds Canvas' 50 MB limit because of MAI_Group5/
(M1 raw dataset + vectors) and the trained joblib artifacts under
models/. The full bundle is mirrored on OneDrive:

    OneDrive URL: <REPLACE_WITH_ONE_DRIVE_LINK_BEFORE_SUBMISSION>

Access is granted to anyone in the @rmit.edu.au tenancy. If the link
is not reachable please contact any group member listed above.
