"""Synthetic data with the credit-card schema.

Used by tests and CI so that the full train -> serve -> score path can run without the Kaggle
dataset. Values are random draws, not real transactions, and must never be reported as results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.data.schema import ALL_COLUMNS, V_COLUMNS


def generate_creditcard_like(n: int = 6000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[: max(2, int(n * fraud_rate))] = 1  # guarantee positives even for tiny n
    rng.shuffle(y)
    v = rng.normal(0.0, 1.0, size=(n, len(V_COLUMNS)))
    # frauds deviate on a handful of components (loosely inspired by V14/V17/V12/V10 in the real data)
    shift = np.zeros(len(V_COLUMNS))
    shift[[13, 16, 11, 9]] = [-4.0, -3.5, -3.0, -2.5]
    shift[[3, 10]] = [2.5, 2.0]
    v[y == 1] += shift + rng.normal(0, 1.0, size=(int(y.sum()), len(V_COLUMNS)))
    amount = np.round(rng.lognormal(3.0, 1.2, size=n), 2)
    t = np.sort(rng.uniform(0, 172_800, size=n)).round()
    df = pd.DataFrame(v, columns=list(V_COLUMNS))
    df.insert(0, "Time", t)
    df["Amount"] = amount
    df["Class"] = y
    return df[list(ALL_COLUMNS)]
