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
recommendations. The recommendation-label predictor for new reviews uses
the Task 3 best model from the notebook: TF-IDF(1,2) review text + title,
plus one-hot product metadata, with LogisticRegression(C=1.0,
class_weight="balanced").

Four tasks, four pages:
- Task 1  pages/browse_and_search.py   Fuzzy keyword search (rapidfuzz)
- Task 2  pages/product_detail.py      Review form + Task 3 best-model label
- Task 3  pages/product_detail.py      Hybrid similar-items recommender
- Task 4  pages/admin_dashboard.py     Admin operations dashboard

Search note (Task 1): the brief mentions matching against "brand name
or description". The Milestone-I dataset has no free-text description
column, so the search field concatenates brand_name + product_title +
product_tags as the closest faithful equivalent — tags carry the
category/ingredient keywords that a description would normally cover.

================================================================================
Task 4 dashboard (HD / real-world use)
================================================================================
The admin dashboard is designed as a practical moderation and customer-recovery
console rather than a static chart page. It supports:

  - sidebar filters for review date, brand, source, final buyer label, and
    minimum risk score;
  - store-health KPIs: review count, buyer rate, average rating, model override
    rate, urgent queue size, and non-buyer share;
  - a risk-scored moderation queue with suggested actions such as buyer
    authenticity checks, customer recovery, and model-disagreement audit;
  - brand and product watchlists that rank operational risk by volume,
    buyer rate, rating, urgent reviews, and average risk score;
  - trend charts for review volume, buyer rate, rating distribution, and brand
    review concentration;
  - model feedback views showing predicted-vs-final labels, confidence bands,
    override rates, and disagreement rows for post-deployment QA.

This makes Task 4 useful for a realistic beauty e-commerce workflow: admins can
identify brands/products needing attention, triage risky reviews, export the
filtered queue, and monitor whether the NLP model stays trustworthy after
new user submissions.

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

3. Train the buyer classifier, sentiment model, legacy comparison models,
   and the Task-3 TF-IDF matrix (one-off; ~30s on a laptop):

       python scripts/train.py

   Writes models/*.joblib, models/product_tfidf_matrix.npz,
   models/metrics.json.

4. Launch the app:

       streamlit run app.py

   Open the URL Streamlit prints (typically http://localhost:8501).

================================================================================
Task 3 best buyer classifier
================================================================================
The active buyer/non-buyer predictor now follows Task 3 from
MAI_Group5/task2_3.ipynb. The notebook's GridSearchCV winner is:

    TF-IDF(1,2) + one-hot metadata
    LogisticRegression(C=1.0, class_weight="balanced")
    Macro-F1 0.7110

Features used:
  - cleaned review text + review title with unigram/bigram TF-IDF;
  - price_log1p, avg_product_rating, rating_count_log1p;
  - one-hot encoded brand_name.

The trained app artifact is models/task3_best_buyer_model.joblib and is
loaded through src/models/loader.py. The older fused artifacts are kept in
models/ for comparison and dashboard context, but Product Detail uses the
Task 3 best model as requested.

See docs/task3_model_summary.md for the marker-facing summary of the source
notebook, features, Macro-F1, artifact path, and label interpretation.

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
  models/                     task3_best_model / sentiment_model /
                              text_model / meta_model / prior_model /
                              fusion / loader
scripts/
  build_catalog.py            Materialises data/products.csv + data/reviews.csv
  train.py                    Trains everything; writes models/*
data/
  products.csv                Generated; one row per product_id
  reviews.csv                 Generated; appended at runtime by Task 2
  cosmetics_beauty_products_reviews.csv
  stopwords_en.txt
models/                       Generated by scripts/train.py
docs/
  task3_model_summary.md
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
- The Task 3 best predictor is loaded lazily as a process-level singleton via
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
