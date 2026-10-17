"""Distribution-shift detection with the Population Stability Index (PSI).

Reference bins are decile edges learned on the *training* split for input features and on
the *validation* split for model scores (training-set scores are in-sample and optimistic).
PSI interpretation bands (0.1 / 0.25) are an industry rule of thumb, not a statistical test;
they are used as warning levels, and small windows are flagged as unreliable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

PSI_WARN = 0.1
PSI_ALERT = 0.25
MIN_WINDOW = 500
_EPS = 1e-4
# Fraud scores are extremely skewed (most mass below 0.1%), so data-driven deciles of validation
# scores put nearly all bins near zero and miss shifts in the tail. Fixed log-spaced probability
# edges keep resolution where alerts happen.
SCORE_EDGES = (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
# Cyclical time-of-day features drift by construction whenever the live window covers only part of
# a day; they are reported but excluded from the overall status.
SEASONAL_FEATURES = frozenset({"hour_sin", "hour_cos"})


def _edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(qs)


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / float(max(1, int(counts.sum())))


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, _EPS, None)
    a = np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(
    features: np.ndarray,
    feature_names: Sequence[str],
    scores: np.ndarray,
    alert_rate: float | None = None,
) -> dict[str, Any]:
    ref: dict[str, Any] = {"features": {}, "n_reference_rows": len(features)}
    for j, name in enumerate(feature_names):
        edges = _edges(features[:, j])
        ref["features"][name] = {
            "edges": edges.tolist(),
            "expected": _proportions(features[:, j], edges).tolist(),
        }
    s_edges = np.asarray(SCORE_EDGES)
    ref["score"] = {
        "edges": s_edges.tolist(),
        "expected": _proportions(scores, s_edges).tolist(),
        "mean": float(np.mean(scores)),
    }
    if alert_rate is not None:
        ref["expected_alert_rate"] = float(alert_rate)
    return ref


def status_for(value: float) -> str:
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARN:
        return "warn"
    return "ok"


def drift_report(
    reference: dict[str, Any],
    features: np.ndarray,
    feature_names: Sequence[str],
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    n = len(features)
    out: dict[str, Any] = {"window_size": n, "reliable": n >= MIN_WINDOW, "features": []}
    for j, name in enumerate(feature_names):
        r = reference["features"].get(name)
        if r is None or n == 0:
            continue
        value = psi(np.asarray(r["expected"]), _proportions(features[:, j], np.asarray(r["edges"])))
        out["features"].append(
            {
                "feature": name,
                "psi": round(value, 4),
                "status": status_for(value),
                "seasonal": name in SEASONAL_FEATURES,
            }
        )
    out["features"].sort(key=lambda d: d["psi"], reverse=True)
    if scores is not None and len(scores) and "score" in reference:
        r = reference["score"]
        value = psi(np.asarray(r["expected"]), _proportions(scores, np.asarray(r["edges"])))
        out["score"] = {
            "psi": round(value, 4),
            "status": status_for(value),
            "mean": float(np.mean(scores)),
            "reference_mean": r.get("mean"),
        }
    core = [f for f in out["features"] if not f["seasonal"]]
    worst = max((f["psi"] for f in core), default=0.0)
    out["max_feature_psi"] = worst
    out["n_features_warn"] = sum(f["status"] == "warn" for f in core)
    out["n_features_alert"] = sum(f["status"] == "alert" for f in core)
    out["seasonal_features_excluded"] = sorted(SEASONAL_FEATURES & set(feature_names))
    out["overall_status"] = (
        status_for(max(worst, out.get("score", {}).get("psi", 0.0)))
        if out["reliable"]
        else "insufficient_data"
    )
    return out
