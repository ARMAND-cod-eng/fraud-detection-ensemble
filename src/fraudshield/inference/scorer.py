"""Online scoring: load a verified artifact and turn raw transactions into scores + context."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fraudshield.data.schema import AMOUNT_COLUMN, RAW_FEATURE_COLUMNS, TIME_COLUMN
from fraudshield.features.engineering import FEATURE_DESCRIPTIONS
from fraudshield.models.artifact import ModelRegistry, load_model, read_json
from fraudshield.models.fraud_model import SIGNAL_DESCRIPTIONS, SIGNAL_NAMES, FraudModel

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
    def __init__(self, model: FraudModel, metadata: dict[str, Any], model_dir: Path) -> None:
        self.model = model
        self.metadata = metadata
        self.model_dir = model_dir
        self.loaded_at = time.time()
        ref_path = model_dir / "reference_profile.json"
        self.reference = read_json(ref_path) if ref_path.exists() else None
        eval_path = model_dir / "evaluation.json"
        self._evaluation_path = eval_path if eval_path.exists() else None
        self._warm_up()

    @classmethod
    def from_registry(cls, artifacts_dir: str | Path, version: str | None = None) -> FraudScorer:
        registry = ModelRegistry(artifacts_dir)
        version = version or registry.production()
        if version is None:
            raise FileNotFoundError(
                f"no production model in {registry.path}; run `fraudshield train --promote` first"
            )
        model_dir = registry.model_dir(version)
        model, meta = load_model(model_dir, verify=True)
        return cls(model, meta, model_dir)

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def review_threshold(self) -> float | None:
        return self.metadata.get("thresholds", {}).get("review")

    def evaluation(self) -> dict[str, Any] | None:
        return read_json(self._evaluation_path) if self._evaluation_path else None

    def _warm_up(self) -> None:
        rec = {c: np.zeros(1) for c in RAW_FEATURE_COLUMNS}
        self.score_columns(rec)

    # ------------------------------------------------------------------ scoring
    def score_columns(self, columns: dict[str, np.ndarray]) -> BatchScores:
        t = time.perf_counter()
        base = self.model.base_features(columns)
        inputs = self.model.model_inputs(base)
        raw = self.model.predict_raw_from_inputs(inputs)
        prob = self.model.calibrator.transform(raw)
        return BatchScores(prob, raw, base, inputs, (time.perf_counter() - t) * 1000)

    @staticmethod
    def records_to_columns(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
        return {
            c: np.fromiter((r[c] for r in records), dtype=np.float64, count=len(records))
            for c in RAW_FEATURE_COLUMNS
        }

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

    def explain(self, inputs_row: np.ndarray, top_k: int = 8) -> dict[str, Any]:
        contrib, base_value = self.model.contributions(inputs_row.reshape(1, -1))
        c = contrib[0]
        names = self.model.model_feature_names
        order = np.argsort(-np.abs(c))[:top_k]
        factors = []
        for i in order:
            name = names[i]
            value = float(inputs_row[i])
            pct = self.model.feature_builder.percentile(name, value)
            if pct is None and name in SIGNAL_NAMES:
                pct = self.model.signal_percentile(name, value)
            factors.append(
                {
                    "feature": name,
                    "value": value,
                    "contribution": float(c[i]),
                    "direction": "increases risk" if c[i] > 0 else "decreases risk",
                    "training_percentile": None if pct is None else round(pct, 2),
                    "description": FEATURE_DESCRIPTIONS.get(name) or SIGNAL_DESCRIPTIONS.get(name, name),
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

    def quality_warnings(self, base_row: np.ndarray, raw: dict[str, float]) -> list[str]:
        """Flag inputs outside the range seen in training (model behaviour there is unvalidated)."""
        warnings: list[str] = []
        prof = self.model.feature_builder.profile
        for j, name in enumerate(self.model.feature_builder.feature_names):
            p = prof.get(name)
            if p and not (p["min"] <= base_row[j] <= p["max"]):
                warnings.append(
                    f"{name}={base_row[j]:.4g} outside training range [{p['min']:.4g}, {p['max']:.4g}]"
                )
        if raw[AMOUNT_COLUMN] == 0:
            warnings.append("Amount is 0 (zero-value authorisation)")
        if raw[TIME_COLUMN] > 172_800:
            warnings.append(
                "Time beyond the 48 h capture window; hour-of-day encoding assumes Time=0 at midnight"
            )
        return warnings[:10]

    def info(self) -> dict[str, Any]:
        m = self.metadata
        return {
            "model_version": self.version,
            "kind": self.model.kind,
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
        }
