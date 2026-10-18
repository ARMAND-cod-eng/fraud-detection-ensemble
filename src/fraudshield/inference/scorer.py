"""Online scoring: load a verified artifact and turn raw transactions into scores + context."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fraudshield.models.artifact import ModelRegistry, load_model, read_json
from fraudshield.models.fraud_model import SIGNAL_DESCRIPTIONS, SIGNAL_NAMES, FraudModel
from fraudshield.profiles.base import ServingProfile
from fraudshield.profiles.creditcard import CreditCardProfile

EXPLANATION_DISCLAIMER = (
    "Attributions (TreeSHAP) show how this model's output depends on each input. They are "
    "statistical associations learned from historical data, not evidence that fraud occurred "
    "and not causal explanations."
)


@dataclass
class BatchScores:
    probability: np.ndarray  # calibrated
    raw: np.ndarray
    base: np.ndarray  # engineered features (for drift / explanations)
    inputs: np.ndarray  # model inputs (base + signals for stacked models)
    model_ms: float


class FraudScorer:
    def __init__(
        self,
        model: FraudModel,
        metadata: dict[str, Any],
        model_dir: Path,
        profile: ServingProfile | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.model_dir = model_dir
        if profile is None:  # offline use (CLI drift report, tests): infer from the artifact
            from fraudshield.profiles.ieee_serving import IEEEProfile

            ieee = getattr(model.feature_builder, "kind", "") == "ieee_v2"
            profile = IEEEProfile(None, None) if ieee else CreditCardProfile(None)
        self.profile = profile
        self.profile.check_model(model)
        self.loaded_at = time.time()
        ref_path = model_dir / "reference_profile.json"
        self.reference = read_json(ref_path) if ref_path.exists() else None
        eval_path = model_dir / "evaluation.json"
        self._evaluation_path = eval_path if eval_path.exists() else None
        self._warm_up()

    @classmethod
    def from_registry(
        cls, artifacts_dir: str | Path, version: str | None = None, profile: ServingProfile | None = None
    ) -> FraudScorer:
        registry = ModelRegistry(artifacts_dir)
        version = version or registry.production()
        if version is None:
            raise FileNotFoundError(
                f"no production model in {registry.path}; run `fraudshield train --promote` first"
            )
        model_dir = registry.model_dir(version)
        model, meta = load_model(model_dir, verify=True)
        return cls(model, meta, model_dir, profile)

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def review_threshold(self) -> float | None:
        return self.metadata.get("thresholds", {}).get("review")

    def evaluation(self) -> dict[str, Any] | None:
        return read_json(self._evaluation_path) if self._evaluation_path else None

    def _warm_up(self) -> None:
        self.score_records([self.profile.warm_up_record()])

    # ------------------------------------------------------------------ scoring
    def _score_base(self, base: np.ndarray, t0: float) -> BatchScores:
        inputs = self.model.model_inputs(base)
        raw = self.model.predict_raw_from_inputs(inputs)
        prob = self.model.calibrator.transform(raw)
        return BatchScores(prob, raw, base, inputs, (time.perf_counter() - t0) * 1000)

    def score_records(self, records: list[dict[str, Any]]) -> BatchScores:
        """Online path: raw records (already enriched by ``profile.prepare``) -> scores."""
        t0 = time.perf_counter()
        return self._score_base(self.profile.base_features(self.model, records), t0)

    def score_columns(self, columns: Any) -> BatchScores:
        """Batch path: a DataFrame or column mapping (offline drift reports, policy simulation)."""
        t0 = time.perf_counter()
        return self._score_base(np.asarray(self.model.base_features(columns)), t0)

    # ------------------------------------------------------------------ context
    def signals(self, base_row: np.ndarray) -> dict[str, Any]:
        """Unsupervised anomaly signals with percentiles vs. validation normals."""
        if not self.model.has_detectors:
            return {}
        values = self.model.signals(base_row.reshape(1, -1))[0]
        out = {}
        for name, v in zip(SIGNAL_NAMES, values, strict=True):
            out[name] = {
                "value": float(v),
                "percentile_vs_normal": self.model.signal_percentile(name, float(v)),
                "used_in_score": self.model.uses_signals,
                "description": SIGNAL_DESCRIPTIONS[name],
            }
        return out

    def explain(
        self, inputs_row: np.ndarray, record: dict[str, Any] | None = None, top_k: int = 8
    ) -> dict[str, Any]:
        contrib, base_value = self.model.contributions(inputs_row.reshape(1, -1))
        c = contrib[0]
        names = self.model.model_feature_names
        order = np.argsort(-np.abs(c))[:top_k]
        factors = []
        for i in order:
            name = names[i]
            value = float(inputs_row[i])
            is_signal = name in SIGNAL_NAMES
            pct = None if np.isnan(value) else self.model.feature_builder.percentile(name, value)
            if pct is None and is_signal:
                pct = self.model.signal_percentile(name, value)
            factors.append(
                {
                    "feature": name,
                    "value": None if np.isnan(value) else value,
                    "display_value": self.profile.display_value(name, value, record),
                    "contribution": float(c[i]),
                    "direction": "increases risk" if c[i] > 0 else "decreases risk",
                    "training_percentile": None if pct is None else round(pct, 2),
                    "description": SIGNAL_DESCRIPTIONS[name]
                    if is_signal
                    else self.profile.feature_description(name),
                    # Plain-language rendering of the same attribution: not a cause, not evidence.
                    "reason": None if is_signal else self.profile.reason(self.model, name, value, record),
                }
            )
        return {
            "method": "TreeSHAP (XGBoost pred_contribs)" if self.model.kind != "logreg" else "linear SHAP",
            "space": "calibrated log-odds",
            "base_value": float(base_value[0]),
            "output_value": float(base_value[0] + c.sum()),
            "top_factors": factors,
            "other_features_contribution": float(c.sum() - c[order].sum()),
            "disclaimer": EXPLANATION_DISCLAIMER,
        }

    @staticmethod
    def top_reasons(explanation: dict[str, Any], k: int = 3) -> list[str]:
        """Risk-increasing factors as sentences, for the API response and the alert queue."""
        return [f["reason"] for f in explanation["top_factors"] if f["contribution"] > 0 and f.get("reason")][
            :k
        ]

    def quality_warnings(self, base_row: np.ndarray, raw: dict[str, Any]) -> list[str]:
        """Flag inputs the model was not validated on (outside training range, unseen categories)."""
        return self.profile.quality_warnings(self.model, base_row, raw)

    def info(self) -> dict[str, Any]:
        m = self.metadata
        return {
            "model_version": self.version,
            "kind": self.model.kind,
            "profile": self.profile.name,
            "selected_candidate": m.get("selected_candidate"),
            "created_at": m.get("created_at"),
            "loaded_at": self.loaded_at,
            "data_sha256": m.get("data_sha256"),
            "split": m.get("split"),
            "feature_names": self.model.feature_builder.feature_names,
            "model_feature_names": self.model.model_feature_names,
            "uses_anomaly_signals_in_score": self.model.uses_signals,
            "auxiliary_signals_available": self.model.has_detectors,
            "calibration": self.model.calibrator.to_dict(),
            "thresholds": m.get("thresholds"),
            "metrics": m.get("metrics"),
            "gate_checks": m.get("gate_checks"),
            "environment": m.get("environment"),
            "online": self.profile.status(),
        }
