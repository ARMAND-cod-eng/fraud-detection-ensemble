"""Risk policy: converts a calibrated fraud probability into a recommended action.

Model prediction and business policy are deliberately separate. The model answers "how likely
is this to be fraud?"; the policy answers "what should we do about it, given costs and review
capacity?". Policies are versioned and every decision records the policy version that made it.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Action = Literal["approve", "review", "decline"]
RiskLevel = Literal["low", "medium", "high"]


class CostAssumptions(BaseModel):
    review_cost: float = Field(default=5.0, ge=0)
    loss_rate: float = Field(default=1.0, ge=0, le=1)
    false_decline_cost: float = Field(default=15.0, ge=0)


class PolicyConfig(BaseModel):
    policy_id: str = "example-balanced"
    description: str = ""
    review_threshold: float = Field(..., gt=0, lt=1)
    decline_threshold: float = Field(0.9, gt=0, le=1)
    expected_loss_review: bool = False
    max_reviews_per_hour: int = Field(120, ge=1)
    cost_assumptions: CostAssumptions = Field(default_factory=lambda: CostAssumptions())

    @model_validator(mode="after")
    def _ordered(self) -> PolicyConfig:
        if self.decline_threshold <= self.review_threshold:
            raise ValueError("decline_threshold must be greater than review_threshold")
        return self

    @property
    def version(self) -> str:
        digest = hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.policy_id}@{digest}"


def load_policy(path: str | Path, model_review_threshold: float | None) -> PolicyConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("review_threshold") == "model":
        if model_review_threshold is None:
            raise ValueError("policy uses the model's review threshold but the model provides none")
        raw["review_threshold"] = float(model_review_threshold)
    return PolicyConfig(**raw)


@dataclass
class Decision:
    action: Action
    risk_level: RiskLevel
    reasons: list[str]
    expected_loss: float
    over_capacity: bool = False


@dataclass
class DecisionEngine:
    policy: PolicyConfig
    _review_times: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history.append(
            {"version": self.policy.version, "at": time.time(), "config": self.policy.model_dump()}
        )

    def update(self, policy: PolicyConfig, actor: str = "api") -> None:
        with self._lock:
            self.policy = policy
            self.history.append(
                {"version": policy.version, "at": time.time(), "actor": actor, "config": policy.model_dump()}
            )

    def reviews_last_hour(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            while self._review_times and self._review_times[0] < now - 3600:
                self._review_times.popleft()
            return len(self._review_times)

    def decide(self, probability: float, amount: float, now: float | None = None) -> Decision:
        p = self.policy
        c = p.cost_assumptions
        expected_loss = probability * amount * c.loss_rate
        reasons: list[str] = []
        if probability >= p.decline_threshold:
            action: Action = "decline"
            risk: RiskLevel = "high"
            reasons.append(
                f"fraud probability {probability:.1%} ≥ decline threshold {p.decline_threshold:.1%}"
            )
        elif probability >= p.review_threshold:
            action, risk = "review", "medium"
            reasons.append(f"fraud probability {probability:.2%} ≥ review threshold {p.review_threshold:.2%}")
        elif p.expected_loss_review and expected_loss >= c.review_cost:
            action, risk = "review", "medium"
            reasons.append(f"expected loss €{expected_loss:.2f} ≥ review cost €{c.review_cost:.2f}")
        else:
            action, risk = "approve", "low"
            reasons.append(
                f"fraud probability {probability:.2%} below review threshold {p.review_threshold:.2%}"
            )

        over = False
        if action == "review":
            now = time.time() if now is None else now
            current = self.reviews_last_hour(now)
            with self._lock:
                self._review_times.append(now)
            if current >= p.max_reviews_per_hour:
                over = True
                reasons.append(
                    f"review capacity exceeded ({current}/{p.max_reviews_per_hour} in the last hour)"
                )
        return Decision(action, risk, reasons, float(expected_loss), over)
