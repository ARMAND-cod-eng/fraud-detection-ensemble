"""Probability calibration.

Platt scaling on the model's log-odds, fit on the validation split. Chosen up front (not by
comparing methods on test) because the validation split holds only a few dozen frauds, where
isotonic regression is prone to overfitting a step function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-12


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, raw_proba: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit(raw_proba).reshape(-1, 1), y)
        self.a, self.b, self.fitted = float(lr.coef_[0, 0]), float(lr.intercept_[0]), True
        return self

    def transform(self, raw_proba: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(raw_proba, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(self.a * logit(raw_proba) + self.b)))

    def to_dict(self) -> dict[str, float | bool | str]:
        return {"method": "platt", "a": self.a, "b": self.b, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, d: dict) -> PlattCalibrator:
        return cls(a=float(d["a"]), b=float(d["b"]), fitted=bool(d["fitted"]))
