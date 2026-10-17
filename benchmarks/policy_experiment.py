"""Decision-policy experiment: how thresholds and cost assumptions change operational cost.

Uses the production model artifact and the same chronological split as training.
* Two-threshold policies (review / decline) are searched on VALIDATION only.
* The chosen policies are then costed once on TEST, next to simple reference policies.
* A sensitivity analysis shows how the cost-optimal review threshold moves with the assumed
  review cost (validation only).

All monetary values rest on illustrative assumptions (configs/decision_policy.yaml); the dataset
contains transaction amounts but no chargeback, review-cost or customer-churn data.

    python benchmarks/policy_experiment.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from fraudshield.data.creditcard import chronological_split, load_creditcard
from fraudshield.decision_engine.policy import CostAssumptions, PolicyConfig, load_policy
from fraudshield.decision_engine.simulate import capacity_curve, simulate_policy
from fraudshield.models.artifact import ModelRegistry, load_model

REVIEW_GRID = np.unique(np.r_[np.geomspace(0.001, 0.8, 40), 0.05, 0.1, 0.25, 0.5])
DECLINE_GRID = np.array([0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])


def policy(
    review: float, decline: float, costs: CostAssumptions, el: bool = False, pid: str = "exp"
) -> PolicyConfig:
    return PolicyConfig(
        policy_id=pid,
        review_threshold=review,
        decline_threshold=decline,
        expected_loss_review=el,
        cost_assumptions=costs,
    )


def best_on(
    y: np.ndarray, p: np.ndarray, amt: np.ndarray, costs: CostAssumptions
) -> tuple[PolicyConfig, dict[str, Any]]:
    best: tuple[float, PolicyConfig, dict[str, Any]] | None = None
    for r, d in itertools.product(REVIEW_GRID, DECLINE_GRID):
        if d <= r:
            continue
        pol = policy(float(r), float(d), costs)
        res = simulate_policy(pol, y, p, amt)
        if best is None or res["cost"]["total"] < best[0]:
            best = (res["cost"]["total"], pol, res)
    assert best is not None
    return best[1], best[2]


def row(name: str, res: dict[str, Any], pol: PolicyConfig | None) -> dict[str, Any]:
    return {
        "policy": name,
        "review_threshold": None if pol is None else pol.review_threshold,
        "decline_threshold": None if pol is None else pol.decline_threshold,
        "approved": res["approved"],
        "reviewed": res["reviewed"],
        "declined": res["declined"],
        "frauds_stopped": f"{res['frauds_stopped']}/{res['frauds']}",
        "legit_declined": res["legit_declined"],
        "fraud_loss": round(res["cost"]["fraud_loss"], 2),
        "review_cost": round(res["cost"]["review"], 2),
        "false_decline_cost": round(res["cost"]["false_declines"], 2),
        "total_cost": round(res["cost"]["total"], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data/raw/creditcard.csv")
    ap.add_argument("--policy", default="configs/decision_policy.yaml")
    ap.add_argument("--out", default="benchmarks/results/policy_experiment")
    args = ap.parse_args()

    reg = ModelRegistry(args.artifacts)
    version = reg.production()
    assert version, "no production model; run `fraudshield train --promote`"
    model, meta = load_model(reg.model_dir(version))
    df, _ = load_creditcard(args.data)
    sc = meta["split_config"]
    split = chronological_split(df, sc["train_frac"], sc["val_frac"])
    yv, ye = split.validation["Class"].to_numpy(), split.test["Class"].to_numpy()
    av, ae = split.validation["Amount"].to_numpy(), split.test["Amount"].to_numpy()
    pv, pe = model.predict_proba(split.validation), model.predict_proba(split.test)
    hours = float(split.test["Time"].max() - split.test["Time"].min()) / 3600

    default = load_policy(args.policy, meta["thresholds"]["review"])
    costs = default.cost_assumptions
    tuned, tuned_val = best_on(yv, pv, av, costs)
    el_policy = policy(
        default.review_threshold, default.decline_threshold, costs, el=True, pid="expected-loss"
    )
    review_only = policy(default.review_threshold, 1.0, costs, pid="review-only")

    named = [
        ("approve everything", policy(0.999, 1.0, costs), True),
        ("example policy (configs/decision_policy.yaml)", default, False),
        ("review-only (never auto-decline)", review_only, False),
        ("example + expected-loss review rule", el_policy, False),
        ("two thresholds tuned on validation", tuned, False),
    ]
    test_rows = []
    for name, pol, approve_all in named:
        r = row(
            name,
            simulate_policy(pol, ye, np.zeros_like(pe) if approve_all else pe, ae),
            None if approve_all else pol,
        )
        r["validation_total_cost"] = round(
            simulate_policy(pol, yv, np.zeros_like(pv) if approve_all else pv, av)["cost"]["total"], 2
        )
        test_rows.append(r)
    sensitivity = []
    for rc in (1.0, 2.5, 5.0, 10.0, 20.0, 50.0):
        c = CostAssumptions(
            review_cost=rc, loss_rate=costs.loss_rate, false_decline_cost=costs.false_decline_cost
        )
        pol, res = best_on(yv, pv, av, c)
        sensitivity.append(
            {
                "review_cost": rc,
                "best_review_threshold": round(pol.review_threshold, 4),
                "best_decline_threshold": pol.decline_threshold,
                "validation_reviews": res["reviewed"],
                "validation_frauds_stopped": f"{res['frauds_stopped']}/{res['frauds']}",
                "validation_total_cost": round(res["cost"]["total"], 2),
            }
        )
    out = {
        "model_version": version,
        "assumptions": costs.model_dump(),
        "note": "Illustrative cost assumptions; thresholds searched on validation, costs reported on test.",
        "test_window_hours": hours,
        "validation_tuned_policy": {
            "review_threshold": tuned.review_threshold,
            "decline_threshold": tuned.decline_threshold,
            "validation_cost": tuned_val["cost"]["total"],
        },
        "test_policies": test_rows,
        "sensitivity_validation": sensitivity,
        "capacity_test": capacity_curve(ye, pe, ae, hours, [5, 10, 25, 50, 100], costs.loss_rate),
        "fraud_amounts_test": {
            "n": int(ye.sum()),
            "median": float(np.median(ae[ye == 1])),
            "share_below_review_cost": float((ae[ye == 1] < costs.review_cost).mean()),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps(out, indent=2))
    md = [
        "# Decision-policy experiment",
        "",
        f"Model `{version}`. {out['note']}",
        f"Assumptions: review €{costs.review_cost}, false decline €{costs.false_decline_cost}, "
        f"loss rate {costs.loss_rate:.0%}. Test window {hours:.1f} h.",
        "",
        "| Policy | Review ≥ | Decline ≥ | Validation total € | Test: reviewed | Test: declined | Test: frauds stopped | Test: legit declined | Test: fraud loss € | Test: review € | Test: friction € | Test total € |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def num(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    for r in test_rows:
        md.append(
            f"| {r['policy']} | {num(r['review_threshold'], '.4f')} | {num(r['decline_threshold'], '.3f')} | "
            f"{r['validation_total_cost']:,.2f} | {r['reviewed']} | {r['declined']} | {r['frauds_stopped']} | {r['legit_declined']} | "
            f"{r['fraud_loss']:,.2f} | {r['review_cost']:,.2f} | {r['false_decline_cost']:,.2f} | {r['total_cost']:,.2f} |"
        )
    md += [
        "",
        "## Sensitivity of the cost-optimal thresholds to review cost (validation)",
        "",
        "| Review cost € | Best review threshold | Best decline threshold | Reviews | Frauds stopped | Cost € |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sensitivity:
        md.append(
            f"| {s['review_cost']} | {s['best_review_threshold']} | {s['best_decline_threshold']} | {s['validation_reviews']} | "
            f"{s['validation_frauds_stopped']} | {s['validation_total_cost']:,.2f} |"
        )
    md += [
        "",
        "## Review capacity (test window, top-scored alerts reviewed first)",
        "",
        "| Reviews / hour | Reviews | Frauds caught | Recall | Precision | Residual loss € |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in out["capacity_test"]:
        md.append(
            f"| {c['reviews_per_hour']} | {c['reviews']} | {c['frauds_caught']} | {c['recall']:.1%} | {c['precision']:.1%} | {c['residual_fraud_loss']:,.2f} |"
        )
    fa = out["fraud_amounts_test"]
    md += [
        "",
        f"Median fraud amount in the test window: €{fa['median']:.2f}; {fa['share_below_review_cost']:.0%} of frauds are "
        f"smaller than the €{costs.review_cost:.0f} review cost, so a pure cost-minimising rule would ignore them. Small "
        "fraudulent amounts are often card-testing that precedes larger fraud, which this cost model does not capture.",
        "",
    ]
    Path(args.out + ".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
