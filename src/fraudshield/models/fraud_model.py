"""The deployable fraud model: features -> (optional anomaly signals) -> classifier -> calibration.

Supported kinds
* ``xgboost`` - gradient-boosted trees on transaction features.
* ``stacked`` - XGBoost on transaction features plus Isolation-Forest and autoencoder scores
  (the original project's ensemble design, trained with out-of-fold detector scores).
* ``logreg``  - standardised logistic regression baseline.

Explanations are exact additive attributions in calibrated log-odds space:
* trees: TreeSHAP computed natively by XGBoost (``pred_contribs=True``), i.e. the same
  algorithm as ``shap.TreeExplainer`` without the extra dependency;
* logistic regression: w_j * z_j, which is the exact SHAP value for a linear model with
  independent features.
Platt calibration is affine in log-odds (a * margin + b), so attributions are rescaled by ``a``
and the base value becomes ``a * base + b``; contributions still sum to the calibrated logit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudshield.features.engineering import FeatureBuilder
from fraudshield.models.anomaly import IsolationForestDetector
from fraudshield.models.autoencoder import AutoencoderDetector
from fraudshield.models.calibration import PlattCalibrator

SIGNAL_NAMES = ("iforest_score", "autoencoder_error")
SIGNAL_DESCRIPTIONS = {
    "iforest_score": "Isolation Forest anomaly score (unsupervised; higher = more isolated)",
    "autoencoder_error": "Autoencoder reconstruction error vs. normal transactions (unsupervised)",
}


@dataclass
class LinearModel:
    """Standardise -> linear -> sigmoid, stored as plain arrays (no pickle)."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def margin(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def contributions(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        return ((x - self.mean) / self.scale) * self.coef, self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LinearModel:
        return cls(
            np.asarray(d["mean"]), np.asarray(d["scale"]), np.asarray(d["coef"]), float(d["intercept"])
        )

    @classmethod
    def from_sklearn(cls, pipeline: Any) -> LinearModel:
        scaler, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
        return cls(scaler.mean_.copy(), scaler.scale_.copy(), lr.coef_[0].copy(), float(lr.intercept_[0]))


def _sigmoid(m: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-m))


@dataclass
class FraudModel:
    kind: str
    feature_builder: FeatureBuilder
    booster: xgb.Booster | None = None
    linear: LinearModel | None = None
    iforest: IsolationForestDetector | None = None
    autoencoder: AutoencoderDetector | None = None
    calibrator: PlattCalibrator = field(default_factory=PlattCalibrator)
    # Training-distribution percentiles of the anomaly signals (for analyst-facing context).
    signal_quantiles: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"xgboost", "stacked", "logreg"}:
            raise ValueError(f"unknown model kind {self.kind!r}")
        if self.kind == "stacked" and (self.iforest is None or self.autoencoder is None):
            raise ValueError("stacked model requires both anomaly detectors")

    # ------------------------------------------------------------------ features & signals
    @property
    def uses_signals(self) -> bool:
        return self.kind == "stacked"

    @property
    def has_detectors(self) -> bool:
        return self.iforest is not None and self.autoencoder is not None

    @property
    def model_feature_names(self) -> list[str]:
        names = list(self.feature_builder.feature_names)
        return names + list(SIGNAL_NAMES) if self.uses_signals else names

    def base_features(self, raw: pd.DataFrame) -> np.ndarray:
        return self.feature_builder.transform(raw)

    def signals(self, base: np.ndarray) -> np.ndarray:
        if not self.has_detectors:
            raise RuntimeError("model has no anomaly detectors")
        assert self.iforest is not None and self.autoencoder is not None
        return np.column_stack([self.iforest.score(base), self.autoencoder.score(base)])

    def model_inputs(self, base: np.ndarray, signals: np.ndarray | None = None) -> np.ndarray:
        if not self.uses_signals:
            return base
        if signals is None:
            signals = self.signals(base)
        return np.hstack([base, signals])

    # ------------------------------------------------------------------ prediction
    def margin(self, x_model: np.ndarray) -> np.ndarray:
        if self.kind == "logreg":
            assert self.linear is not None
            return self.linear.margin(x_model)
        assert self.booster is not None
        return np.asarray(self.booster.inplace_predict(x_model, predict_type="margin"), dtype=np.float64)

    def predict_raw_from_inputs(self, x_model: np.ndarray) -> np.ndarray:
        return _sigmoid(self.margin(x_model))

    def predict_from_inputs(self, x_model: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw_from_inputs(x_model))

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.predict_from_inputs(self.model_inputs(self.base_features(raw)))

    # ------------------------------------------------------------------ explanations
    def contributions(self, x_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature attributions and base values in *calibrated* log-odds space."""
        if self.kind == "logreg":
            assert self.linear is not None
            contrib, base = self.linear.contributions(x_model)
            base_arr = np.full(len(x_model), base)
        else:
            assert self.booster is not None
            out = self.booster.predict(xgb.DMatrix(x_model), pred_contribs=True)
            contrib, base_arr = out[:, :-1], out[:, -1]
        a, b = (self.calibrator.a, self.calibrator.b) if self.calibrator.fitted else (1.0, 0.0)
        return contrib * a, base_arr * a + b

    def prepare_for_serving(self, xgb_threads: int = 1) -> FraudModel:
        """Single-transaction scoring is latency-bound: thread pools cost more than they save."""
        if self.booster is not None:
            self.booster.set_param({"nthread": xgb_threads})
        if self.iforest is not None and self.iforest.model_ is not None:
            self.iforest.model_.set_params(n_jobs=1)
        return self

    def signal_percentile(self, name: str, value: float) -> float | None:
        q = self.signal_quantiles.get(name)
        if not q:
            return None
        return float(np.interp(value, np.asarray(q), np.linspace(0, 100, len(q))))
