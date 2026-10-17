from __future__ import annotations

import numpy as np
import pytest

from fraudshield.decision_engine.policy import DecisionEngine, PolicyConfig, load_policy
from fraudshield.decision_engine.simulate import capacity_curve, simulate_policy
from fraudshield.monitoring.drift import build_reference, drift_report, psi
from fraudshield.training import evaluate as ev


# ----------------------------------------------------------------------------- evaluation
def test_threshold_metrics_hand_computed() -> None:
    y = np.array([1, 1, 0, 0, 0, 1])
    s = np.array([0.9, 0.4, 0.8, 0.1, 0.2, 0.7])
    m = ev.threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_recall_at_fpr_and_precision_at_k() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.95, 0.9, 0.8, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02])
    assert ev.recall_at_fpr(y, s, 0.0) == 0.5  # only the top-scored fraud before the first FP
    assert ev.recall_at_fpr(y, s, 1 / 8) == 1.0
    assert ev.precision_at_k(y, s, 2) == 0.5
    assert ev.precision_at_k(y, s, 3) == pytest.approx(2 / 3)


def test_best_f1_threshold_on_separable_data() -> None:
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = ev.best_f1_threshold(y, s)
    assert ev.threshold_metrics(y, s, t)["f1"] == 1.0


def test_bootstrap_ci_contains_point_and_paired_delta() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    good = y + rng.normal(0, 0.5, 3000)
    bad = y + rng.normal(0, 2.0, 3000)
    out = ev.bootstrap(y, {"good": good, "bad": bad}, ev.roc_auc_score, n_boot=200, reference="good")
    for k in out:
        assert out[k]["ci_low"] <= out[k]["point"] <= out[k]["ci_high"]
    assert out["bad"]["delta_ci_high"] < 0  # 'bad' is reliably worse
    assert out["bad"]["p_delta_le_0"] > 0.95


def test_review_cost_curve_extremes() -> None:
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.2, 0.6])
    amt = np.array([100.0, 10.0, 10.0, 50.0])
    curve = ev.review_cost_curve(y, p, amt, review_cost=5.0, thresholds=np.array([0.0, 0.5, 0.95]))
    assert curve[0]["total_cost"] == 4 * 5.0  # review everything
    assert curve[1]["total_cost"] == 2 * 5.0  # catch both frauds with two reviews
    assert curve[2]["total_cost"] == 150.0  # review nothing: lose both frauds


def test_calibration_metrics_perfect_vs_bad() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    good = ev.calibration_metrics(y, p)
    bad = ev.calibration_metrics(y, np.clip(p * 0.3, 0, 1))
    assert good["ece"] < 0.02 < bad["ece"]
    assert good["brier"] < bad["brier"]


# ----------------------------------------------------------------------------- policy
def _policy(**kw) -> PolicyConfig:
    base = {"review_threshold": 0.1, "decline_threshold": 0.9, "max_reviews_per_hour": 2}
    return PolicyConfig(**{**base, **kw})


def test_decision_bands() -> None:
    eng = DecisionEngine(_policy())
    assert eng.decide(0.05, 10).action == "approve"
    assert eng.decide(0.5, 10).action == "review"
    d = eng.decide(0.95, 10)
    assert (d.action, d.risk_level) == ("decline", "high")
    assert d.expected_loss == pytest.approx(9.5)


def test_expected_loss_rule_and_capacity_flag() -> None:
    eng = DecisionEngine(_policy(expected_loss_review=True))
    assert eng.decide(0.02, 1000.0, now=0.0).action == "review"  # 0.02 * 1000 >= 5
    assert eng.decide(0.02, 10.0, now=1.0).action == "approve"
    assert not eng.decide(0.5, 10, now=2.0).over_capacity  # 2nd review this hour
    flagged = eng.decide(0.5, 10, now=3.0)
    assert flagged.over_capacity and "capacity" in flagged.reasons[-1]
    assert not eng.decide(0.5, 10, now=4000.0).over_capacity  # window has rolled


def test_policy_validation_and_versioning(tmp_path) -> None:
    with pytest.raises(ValueError):
        PolicyConfig(review_threshold=0.5, decline_threshold=0.4)
    assert _policy().version != _policy(review_threshold=0.2).version
    f = tmp_path / "p.yaml"
    f.write_text("policy_id: t\nreview_threshold: model\ndecline_threshold: 0.9\n")
    assert load_policy(f, 0.05).review_threshold == 0.05
    with pytest.raises(ValueError):
        load_policy(f, None)


def test_simulate_policy_costs() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.95, 0.05, 0.5, 0.95])
    amt = np.array([100.0, 40.0, 20.0, 30.0])
    r = simulate_policy(_policy(), y, p, amt)
    assert (r["approved"], r["reviewed"], r["declined"]) == (1, 1, 2)
    assert r["cost"]["fraud_loss"] == 40.0
    assert r["cost"]["review"] == 5.0
    assert r["cost"]["false_declines"] == 15.0
    caps = capacity_curve(y, p, amt, hours=1.0, capacities=[1, 4])
    assert caps[0]["reviews"] == 1 and caps[1]["recall"] == 1.0


# ----------------------------------------------------------------------------- drift
def test_psi_stable_vs_shifted() -> None:
    rng = np.random.default_rng(0)
    ref_x = rng.normal(size=(20_000, 2))
    ref = build_reference(ref_x, ["a", "b"], rng.uniform(size=20_000))
    same = drift_report(ref, rng.normal(size=(5000, 2)), ["a", "b"], rng.uniform(size=5000))
    shifted = drift_report(
        ref, rng.normal(size=(5000, 2)) + np.array([1.0, 0.0]), ["a", "b"], rng.uniform(size=5000) ** 3
    )
    assert same["overall_status"] == "ok"
    assert shifted["features"][0]["feature"] == "a"
    assert shifted["features"][0]["status"] == "alert"
    assert shifted["score"]["status"] == "alert"
    small = drift_report(ref, rng.normal(size=(50, 2)), ["a", "b"])
    assert small["overall_status"] == "insufficient_data"
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
