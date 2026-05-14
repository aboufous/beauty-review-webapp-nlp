# Task 3 Model Summary

This web app uses the best buyer/non-buyer classifier from Milestone 1
Task 3 as the active prediction model on the Product Detail page.

## Source Notebook

Original notebook:

`/MAI_Group5/task2_3.ipynb`

The notebook compares review classification models for the target label
`is_a_buyer`, using 5-fold stratified cross-validation and Macro F1 as the
main metric because the buyer/non-buyer classes are imbalanced.

## Best Model

The best model selected from Task 3 is:

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

## Feature Set

The active app model uses:

- cleaned `review_text` + `review_title`
- TF-IDF unigram and bigram features
- `price_log1p`
- `avg_product_rating`
- `rating_count_log1p`
- one-hot encoded `brand_name`

This matches the notebook's improved Task 3 feature representation:

`TF-IDF (1,2) + One-Hot Extra`

## App Artifact

The trained artifact used by the web app is:

`models/task3_best_buyer_model.joblib`

The loader is:

`src/models/loader.py`

The model implementation is:

`src/models/task3_best_model.py`

The Product Detail page calls `load_predictor()` and uses this model for the
verified-buyer prediction.

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
