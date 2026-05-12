"""Bayesian-smoothed prior on is_a_buyer per product (falls back to brand, global).

Pure pandas; no sklearn. The "model" is just three dicts persisted via joblib.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ALPHA_PRODUCT = 10.0  # pseudo-count for product-level smoothing toward brand rate
ALPHA_BRAND = 5.0     # pseudo-count for brand-level smoothing toward global rate


@dataclass
class PriorModel:
    global_rate: float = 0.5
    brand_rates: dict = field(default_factory=dict)
    product_rates: dict = field(default_factory=dict)

    def predict_proba_one(self, product_id, brand_name) -> float:
        pid = str(product_id) if product_id is not None else None
        bn = brand_name if pd.notna(brand_name) else None
        if pid is not None and pid in self.product_rates:
            return float(self.product_rates[pid])
        if bn is not None and bn in self.brand_rates:
            return float(self.brand_rates[bn])
        return float(self.global_rate)


def fit(df: pd.DataFrame) -> PriorModel:
    """Fit smoothed priors on a training DataFrame containing is_a_buyer."""
    y = df["is_a_buyer"].astype("float")
    global_rate = float(y.mean())

    brand = df.assign(_y=y).groupby("brand_name", dropna=False)["_y"].agg(["sum", "count"])
    brand_rates = (
        (brand["sum"] + ALPHA_BRAND * global_rate) / (brand["count"] + ALPHA_BRAND)
    ).to_dict()

    # Smooth product rate toward its brand rate.
    prod = df.assign(_y=y).groupby(["product_id", "brand_name"], dropna=False)["_y"].agg(["sum", "count"]).reset_index()
    prod["brand_rate"] = prod["brand_name"].map(brand_rates).fillna(global_rate)
    prod["rate"] = (prod["sum"] + ALPHA_PRODUCT * prod["brand_rate"]) / (prod["count"] + ALPHA_PRODUCT)
    product_rates = {str(pid): float(r) for pid, r in zip(prod["product_id"], prod["rate"])}

    return PriorModel(
        global_rate=global_rate,
        brand_rates={k: float(v) for k, v in brand_rates.items() if pd.notna(k)},
        product_rates=product_rates,
    )
