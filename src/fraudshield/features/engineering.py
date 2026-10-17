"""Feature construction shared by training and online inference.

Design rules (they fix the defects found in the legacy notebook):
* Every engineered feature is a function of a single transaction, so the batch path and the
  single-row path are the same code and cannot drift apart (legacy train/serve skew).
* ``fit`` learns nothing that ``transform`` uses. It only records a training profile that is
  used for data-quality checks and for percentile context in explanations. There is no
  statistic fitted on test data (legacy leakage).
* The raw ``Time`` column (seconds since the first transaction in the 48 h capture) is not used
  as a model input: it encodes position in the dataset, not a property of the transaction, and
  is out of range for any future data. Only a cyclical hour-of-day encoding is derived from it,
  under the documented assumption that ``Time = 0`` is the start of a day.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.data.schema import AMOUNT_COLUMN, TIME_COLUMN, V_COLUMNS

ENGINEERED_FEATURES: tuple[str, ...] = (
    "amount_log1p",
    "hour_sin",
    "hour_cos",
    "v_l2_norm",
    "v_abs_max",
    "v_mean",
    "v_std",
)
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "Amount": "Transaction amount (EUR)",
    "amount_log1p": "log(1 + amount)",
    "hour_sin": "Hour-of-day, cyclical (sine); assumes Time=0 is midnight",
    "hour_cos": "Hour-of-day, cyclical (cosine); assumes Time=0 is midnight",
    "v_l2_norm": "Euclidean norm of the 28 PCA components (overall atypicality)",
    "v_abs_max": "Largest absolute PCA component value",
    "v_mean": "Mean of the 28 PCA components",
    "v_std": "Standard deviation of the 28 PCA components",
    **{v: f"Anonymised PCA component {v} (meaning not disclosed by the data provider)" for v in V_COLUMNS},
}
_QUANTILE_LEVELS = np.linspace(0.0, 1.0, 201)


class FeatureBuilder:
    version = "fb-2"

    def __init__(self, engineered: bool = True) -> None:
        self.engineered = engineered
        self.feature_names: list[str] = [*V_COLUMNS, AMOUNT_COLUMN]
        if engineered:
            self.feature_names += list(ENGINEERED_FEATURES)
        self.profile: dict[str, dict[str, Any]] = {}

    @property
    def raw_columns(self) -> list[str]:
        return [TIME_COLUMN, *V_COLUMNS, AMOUNT_COLUMN]

    def transform(self, data: pd.DataFrame | Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
        """Raw columns (Time, V1..V28, Amount) -> model feature matrix.

        Accepts a DataFrame or any mapping of column -> 1-D array (the API passes a dict to
        avoid DataFrame construction on the single-row hot path).
        """

        def col(name: str) -> np.ndarray:
            return np.asarray(data[name], dtype=np.float64).reshape(-1)

        v = np.column_stack([col(c) for c in V_COLUMNS])
        amount = col(AMOUNT_COLUMN)
        parts = [v, amount[:, None]]
        if self.engineered:
            hour = (col(TIME_COLUMN) / 3600.0) % 24.0
            angle = 2.0 * np.pi * hour / 24.0
            parts.append(
                np.column_stack(
                    [
                        np.log1p(np.clip(amount, 0.0, None)),
                        np.sin(angle),
                        np.cos(angle),
                        np.sqrt((v**2).sum(axis=1)),
                        np.abs(v).max(axis=1),
                        v.mean(axis=1),
                        v.std(axis=1),  # population std (ddof=0) in both batch and single-row use
                    ]
                )
            )
        return np.hstack(parts)

    def fit(self, data: pd.DataFrame) -> FeatureBuilder:
        x = self.transform(data)
        self.profile = {}
        for j, name in enumerate(self.feature_names):
            col = x[:, j]
            q = np.quantile(col, _QUANTILE_LEVELS)
            self.profile[name] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "p001": float(np.quantile(col, 0.001)),
                "p999": float(np.quantile(col, 0.999)),
                "quantiles": [float(val) for val in q],
            }
        return self

    def percentile(self, name: str, value: float) -> float | None:
        """Approximate training-distribution percentile (0-100) of ``value`` for a feature."""
        prof = self.profile.get(name)
        if not prof:
            return None
        q = np.asarray(prof["quantiles"])
        return float(np.interp(value, q, _QUANTILE_LEVELS * 100.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "engineered": self.engineered,
            "feature_names": self.feature_names,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureBuilder:
        fb = cls(engineered=bool(payload["engineered"]))
        if list(payload["feature_names"]) != fb.feature_names:
            raise ValueError("feature_names in artifact do not match this FeatureBuilder version")
        fb.profile = dict(payload.get("profile", {}))
        return fb
