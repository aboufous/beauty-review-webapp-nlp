# Task 3 And Final Fusion Model Summary

This web app keeps the best buyer/non-buyer classifier from Milestone 1
Task 3 as a benchmark, then uses a fused model as the active final
prediction on the Product Detail page to satisfy the DI/HD requirement.

## Source Notebook

Original notebook:

`/MAI_Group5/task2_3.ipynb`

The notebook compares review classification models for the target label
`is_a_buyer`, using 5-fold stratified cross-validation and Macro F1 as the
main metric because the buyer/non-buyer classes are imbalanced.

## Task 3 Best Single Model

The best single model selected from Task 3 is:

`TF-IDF(1,2) + one-hot metadata + LogisticRegression`

GridSearchCV best parameters:

```text
C = 1.0
class_weight = "balanced"
solver = "liblinear"
```

Notebook result:

```text
Macro F1 = 0.7110
```

## Task 3 Benchmark Feature Set

The Task 3 benchmark model uses:

- cleaned `review_text` + `review_title`
- TF-IDF unigram and bigram features
- `price_log1p`
- `avg_product_rating`
- `rating_count_log1p`
- one-hot encoded `brand_name`

This matches the notebook's improved Task 3 feature representation:

`TF-IDF (1,2) + One-Hot Extra`

## Final App Model

The final app prediction uses a three-signal fusion model:

- text model: TF-IDF review title/body classifier
- metadata model: review rating + product metadata classifier
- prior model: Bayesian-smoothed product/brand history

These independently trained models use different data types and are fused by a
LogisticRegression stacker. This is the active Product Detail prediction and
matches the DI/HD requirement for a fused final result.

Held-out test metrics:

```text
text   Macro-F1 0.627   acc 0.706
meta   Macro-F1 0.715   acc 0.834
prior  Macro-F1 0.676   acc 0.818
fused  Macro-F1 0.741   acc 0.841
```

Active app artifacts:

```text
models/text_pipeline.joblib
models/meta_bundle.joblib
models/prior_model.joblib
models/fusion.joblib
```

The Task 3 benchmark artifact is retained for traceability:

`models/task3_best_buyer_model.joblib`

The loader is:

`src/models/loader.py`

The Task 3 benchmark implementation is:

`src/models/task3_best_model.py`

The Product Detail page calls `load_predictor()` from `src/models/loader.py`,
which loads the fused predictor as the active final model.

## Important Interpretation

The predicted label answers:

> Does this review read like it came from a verified buyer?

It does not answer:

> Does the customer want to buy again?

For this reason, the Product Detail page shows two separate outputs:

- **Verified-buyer prediction**: buyer authenticity / purchase likelihood signal
- **Customer sentiment**: positive, negative, or mixed satisfaction signal

A review can therefore be:

- likely verified buyer + positive sentiment
- likely verified buyer + negative sentiment
- non-buyer / unverified + positive sentiment
- non-buyer / unverified + negative sentiment

This separation avoids confusing `buyer` with `repurchase intent`.
