"""Offline policy simulation on labelled data (used by the policy experiment and the dashboard)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fraudshield.decision_engine.policy import PolicyConfig


def simulate_policy(policy: PolicyConfig, y: np.ndarray, p: np.ndarray, amount: np.ndarray) -> dict[str, Any]:
    """Apply a two-threshold policy to labelled transactions and cost it.

    Cost model (illustrative, see configs/decision_policy.yaml):
      approve & fraud   -> loss_rate * amount
      review            -> review_cost (reviewed fraud is assumed caught)
      decline & legit   -> false_decline_cost
      decline & fraud   -> 0
    Review capacity is not simulated here; see ``capacity_curve`` for that trade-off.
    """
    c = policy.cost_assumptions
    decline = p >= policy.decline_threshold
    review = ~decline & (p >= policy.review_threshold)
    if policy.expected_loss_review:
        review |= ~decline & (p * amount * c.loss_rate >= c.review_cost)
    approve = ~decline & ~review
    fraud = y == 1
    fraud_loss = float(amount[approve & fraud].sum() * c.loss_rate)
    review_cost = float(review.sum() * c.review_cost)
    friction = float((decline & ~fraud).sum() * c.false_decline_cost)
    return {
        "policy_version": policy.version,
        "transactions": len(y),
        "approved": int(approve.sum()),
        "reviewed": int(review.sum()),
        "declined": int(decline.sum()),
        "frauds": int(fraud.sum()),
        "frauds_stopped": int(((review | decline) & fraud).sum()),
        "fraud_amount_total": float(amount[fraud].sum()),
        "fraud_amount_stopped": float(amount[(review | decline) & fraud].sum()),
        "legit_declined": int((decline & ~fraud).sum()),
        "legit_reviewed": int((review & ~fraud).sum()),
        "cost": {
            "fraud_loss": fraud_loss,
            "review": review_cost,
            "false_declines": friction,
            "total": fraud_loss + review_cost + friction,
        },
    }


def capacity_curve(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    hours: float,
    capacities: list[int],
    loss_rate: float = 1.0,
) -> list[dict[str, float]]:
    """If analysts can only review C alerts per hour, review the top-scored C*hours alerts.

    Returns fraud caught and residual loss per capacity level - the 'precision at a fixed
    investigation budget' view of the same model.
    """
    order = np.argsort(-p)
    fraud = y == 1
    out = []
    for cap in capacities:
        k = min(len(y), round(cap * hours))
        top = order[:k]
        caught = int(fraud[top].sum())
        out.append(
            {
                "reviews_per_hour": cap,
                "reviews": k,
                "frauds_caught": caught,
                "recall": caught / max(1, int(fraud.sum())),
                "precision": caught / max(1, k),
                "residual_fraud_loss": float(amount[fraud].sum() - amount[top][fraud[top]].sum()) * loss_rate,
            }
        )
    return out
