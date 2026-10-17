"""Evaluation-gated promotion and rollback.

A model never becomes the serving model just because it was trained. ``promote_version``
re-verifies the artifact, smoke-tests it, checks the pre-declared gate thresholds and compares
against the current production model (only when both were evaluated on the same data).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraudshield.data.synthetic import generate_creditcard_like
from fraudshield.models.artifact import ModelRegistry, load_model, read_json, verify_artifact


class PromotionRejected(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__(
            "promotion gate failed: " + "; ".join(c["name"] for c in report["checks"] if not c["passed"])
        )
        self.report = report


def evaluate_gate(registry: ModelRegistry, version: str, gate: dict[str, Any]) -> dict[str, Any]:
    d = registry.model_dir(version)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        verify_artifact(d)
        check("artifact_integrity", True, "all SHA-256 checksums match")
    except Exception as exc:  # report, don't raise
        check("artifact_integrity", False, str(exc))
        return {"version": version, "passed": False, "checks": checks}

    model, meta = load_model(d)
    probe = generate_creditcard_like(n=64, seed=123)
    p1, p2 = model.predict_proba(probe), model.predict_proba(probe)
    check(
        "smoke_test",
        bool(np.isfinite(p1).all() and ((p1 >= 0) & (p1 <= 1)).all() and np.array_equal(p1, p2)),
        "finite, in [0, 1], deterministic on 64 synthetic probe rows",
    )

    # Measured values come from the artifact; limits come from the gate config in force *now*, so
    # tightening the gate after training applies to every later promotion.
    current_limits = {
        "test_pr_auc": gate.get("min_test_pr_auc"),
        "test_roc_auc": gate.get("min_test_roc_auc"),
        "p95_latency_ms": gate.get("max_p95_latency_ms"),
    }
    for name, (value, op, recorded_limit, _) in meta.get("gate_checks", {}).items():
        limit = current_limits.get(name)
        limit = recorded_limit if limit is None else limit
        ok = value >= limit if op == ">=" else value <= limit
        check(name, ok, {"value": value, "op": op, "limit": limit})

    current = registry.production()
    if current and current != version:
        cur_meta = read_json(registry.model_dir(current) / "metadata.json")
        comparable = cur_meta.get("data_sha256") == meta.get("data_sha256") and cur_meta.get(
            "split"
        ) == meta.get("split")
        if comparable:
            delta = meta["metrics"]["test"]["pr_auc"] - cur_meta["metrics"]["test"]["pr_auc"]
            check(
                "no_regression_vs_production",
                delta >= -gate["max_pr_auc_regression"],
                {"production": current, "pr_auc_delta": delta, "tolerance": gate["max_pr_auc_regression"]},
            )
        else:
            check(
                "no_regression_vs_production",
                False,
                {
                    "production": current,
                    "detail": "not comparable (different data or split); use --force after review",
                },
            )
    return {"version": version, "passed": all(c["passed"] for c in checks), "checks": checks}


def promote_version(
    registry: ModelRegistry, version: str, gate: dict[str, Any], reason: str = "", force: bool = False
) -> dict[str, Any]:
    report = evaluate_gate(registry, version, gate)
    if not report["passed"] and not force:
        raise PromotionRejected(report)
    registry.set_production(version, "promote", reason or ("forced" if force else "gate passed"), gate=report)
    return report
