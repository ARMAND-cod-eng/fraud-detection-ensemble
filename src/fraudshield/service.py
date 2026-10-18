"""Scoring service: validation -> idempotency -> online features -> model -> policy -> persistence.

The HTTP API and the replay engine both call ``FraudService.score``, so replayed traffic
exercises exactly the production code path. Dataset-specific steps (field contract, online
features, entity links, analyst text) come from the scorer's serving profile.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fraudshield.decision_engine.policy import DecisionEngine
from fraudshield.inference.scorer import FraudScorer
from fraudshield.monitoring.live import LiveMonitor
from fraudshield.monitoring.metrics import Metrics
from fraudshield.storage.store import PredictionStore
from fraudshield.streaming.broadcaster import Broadcaster


class IdempotencyConflict(Exception):
    def __init__(self, transaction_id: str) -> None:
        super().__init__(transaction_id)
        self.transaction_id = transaction_id


class ExplanationUnavailable(Exception):
    pass


@dataclass
class ScoreRequest:
    features: dict[str, Any]  # raw fields of the active profile's contract
    transaction_id: str | None = None
    ground_truth: int | None = None  # only for replayed labelled history; never shown to the model
    synthetic: bool = False


class FraudService:
    def __init__(
        self,
        scorer: FraudScorer,
        engine: DecisionEngine,
        store: PredictionStore,
        monitor: LiveMonitor,
        metrics: Metrics,
        broadcaster: Broadcaster,
        max_stored: int = 200_000,
    ) -> None:
        self.scorer = scorer
        self.profile = scorer.profile
        self.engine = engine
        self.store = store
        self.monitor = monitor
        self.metrics = metrics
        self.broadcaster = broadcaster
        self.max_stored = max_stored
        self._writes_since_prune = 0
        metrics.model_info.labels(scorer.version, scorer.model.kind).set(1)

    def score(
        self, requests: list[ScoreRequest], source: str = "api", scenario: str | None = None
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        results: list[dict[str, Any] | None] = [None] * len(requests)
        new_idx: list[int] = []
        hashes: list[str] = []
        ids: list[str] = []
        for i, req in enumerate(requests):
            tid = req.transaction_id or f"txn_{uuid.uuid4().hex[:20]}"
            h = self.profile.payload_hash(req.features)
            ids.append(tid)
            hashes.append(h)
            if req.transaction_id:
                existing = self.store.get_hash(tid)
                if existing is not None:
                    if existing != h:
                        raise IdempotencyConflict(tid)
                    stored = self.store.get(tid)
                    assert stored is not None
                    results[i] = {**self._response(stored), "idempotent_replay": True}
                    continue
            new_idx.append(i)

        if new_idx:
            # Online features read state *before* each event and then record it; only new
            # transaction IDs get here, so idempotent retries never double-count.
            t_feat = time.perf_counter()
            records, online_warnings = self.profile.prepare([requests[i].features for i in new_idx])
            feature_ms = (time.perf_counter() - t_feat) * 1000
            scores = self.scorer.score_records(records)
            per_txn_ms = (scores.model_ms + feature_ms) / len(new_idx)
            now = time.time()
            rows, links, alerts = [], [], 0
            for k, i in enumerate(new_idx):
                req, rec = requests[i], records[k]
                p, raw = float(scores.probability[k]), float(scores.raw[k])
                amount = float(rec[self.profile.amount_field])
                decision = self.engine.decide(p, amount, now)
                warnings = online_warnings[k] + self.scorer.quality_warnings(scores.base[k], rec)
                explanation = signals = None
                top_reasons: list[str] = []
                if decision.action != "approve":
                    alerts += 1
                    explanation = self.scorer.explain(scores.inputs[k], rec)
                    signals = self.scorer.signals(scores.base[k])
                    top_reasons = self.scorer.top_reasons(explanation)
                event_time = float(rec[self.profile.time_field])
                row = {
                    "transaction_id": ids[i],
                    "payload_hash": hashes[i],
                    "received_at": now,
                    "event_time": event_time,
                    "amount": amount,
                    "fraud_probability": p,
                    "raw_score": raw,
                    "risk_level": decision.risk_level,
                    "action": decision.action,
                    "reasons": decision.reasons,
                    "expected_loss": decision.expected_loss,
                    "over_capacity": int(decision.over_capacity),
                    "model_version": self.scorer.version,
                    "policy_version": self.engine.policy.version,
                    "latency_ms": per_txn_ms,
                    "source": source,
                    "synthetic": int(req.synthetic),
                    "scenario": scenario,
                    "ground_truth": req.ground_truth,
                    "features": self.profile.stored_features(rec),
                    "explanation": explanation,
                    "signals": signals,
                    "quality": warnings[:10],
                    "review_status": "open" if decision.action != "approve" else "none",
                    "top_reasons": top_reasons,
                }
                rows.append(row)
                links += [(kind, h, ids[i], event_time) for kind, h in self.profile.entity_keys(rec).items()]
                self.monitor.record_quality(warnings)
                if warnings:
                    self.metrics.quality_warnings.inc(len(warnings))
                self.metrics.predictions.labels(decision.action, source, self.scorer.version).inc()
                self.metrics.score.observe(p)
            self.store.insert_predictions(rows, links)
            self.monitor.record_scores(scores.base, scores.probability, alerts, scores.model_ms)
            self.metrics.inference_latency.observe(per_txn_ms / 1000)
            self.metrics.reviews_last_hour.set(self.engine.reviews_last_hour())
            for k, i in enumerate(new_idx):
                resp = {**self._response(rows[k]), "idempotent_replay": False}
                results[i] = resp
                self.broadcaster.publish({"type": "transaction", "data": self._event(rows[k])})
            self._maybe_prune(len(rows))
        out = [r for r in results if r is not None]
        for r in out:
            r.setdefault("processing_ms", (time.perf_counter() - t0) * 1000 / len(requests))
        return out

    def _maybe_prune(self, n: int) -> None:
        self._writes_since_prune += n
        if self._writes_since_prune >= 5000:
            self._writes_since_prune = 0
            self.store.prune(self.max_stored)

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds")

    def _response(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "transaction_id": row["transaction_id"],
            "fraud_probability": row["fraud_probability"],
            "raw_model_score": row["raw_score"],
            "risk_level": row["risk_level"],
            "recommended_action": row["action"],
            "decision_reasons": row["reasons"],
            "top_risk_factors": row.get("top_reasons") or [],
            "expected_loss_eur": row["expected_loss"],
            "review_capacity_exceeded": bool(row["over_capacity"]),
            "model_version": row["model_version"],
            "policy_version": row["policy_version"],
            "scored_at": self._iso(row["received_at"]),
            "inference_ms": row["latency_ms"],
            "explanation_available": True,
            "explanation_url": f"/api/v1/transactions/{row['transaction_id']}/explanation",
            "data_quality_warnings": row["quality"],
            "synthetic": bool(row["synthetic"]),
        }

    def _event(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "transaction_id": row["transaction_id"],
            "received_at": row["received_at"],
            "amount": row["amount"],
            "fraud_probability": row["fraud_probability"],
            "risk_level": row["risk_level"],
            "action": row["action"],
            "source": row["source"],
            "synthetic": bool(row["synthetic"]),
            "scenario": row["scenario"],
            "latency_ms": row["latency_ms"],
            "review_status": row["review_status"],
            "top_reason": (row.get("top_reasons") or [None])[0],
        }

    # ------------------------------------------------------------------ explanations
    def explanation(self, transaction_id: str) -> dict[str, Any] | None:
        row = self.store.get(transaction_id)
        if row is None:
            return None
        if row.get("explanation") is not None:
            return {
                "transaction_id": transaction_id,
                "model_version": row["model_version"],
                "explanation": row["explanation"],
                "signals": row.get("signals") or {},
                "cached": True,
            }
        if row["model_version"] != self.scorer.version:
            raise ExplanationUnavailable(
                f"scored by model {row['model_version']}; current model is {self.scorer.version}"
            )
        # Stored features include the online features as they were at scoring time, so this
        # recomputation is point-in-time and does not touch the online store.
        rec = {k: (float("nan") if v is None else v) for k, v in row["features"].items()}
        scores = self.scorer.score_records([rec])
        explanation = self.scorer.explain(scores.inputs[0], rec)
        signals = self.scorer.signals(scores.base[0])
        self.store.set_explanation(transaction_id, explanation, signals)
        return {
            "transaction_id": transaction_id,
            "model_version": row["model_version"],
            "explanation": explanation,
            "signals": signals,
            "cached": False,
        }

    # ------------------------------------------------------------------ linked activity
    def linked(self, transaction_id: str, limit: int = 10) -> dict[str, Any] | None:
        row = self.store.get(transaction_id)
        if row is None:
            return None
        if not self.profile.linked_activity:
            return {
                "transaction_id": transaction_id,
                "available": False,
                "reason": f"the {self.profile.name} data has no card, customer or device identifiers",
                "links": [],
            }
        labels = self.profile.link_labels()
        links = []
        for kind, summary in self.store.linked(transaction_id, row["event_time"], limit).items():
            label, strength = labels.get(kind, (kind, "unknown"))
            links.append({"type": kind, "label": label, "strength": strength, **summary})
        order = list(labels)
        links.sort(key=lambda d: order.index(d["type"]) if d["type"] in order else len(order))
        return {
            "transaction_id": transaction_id,
            "available": True,
            "links": links,
            "note": "Transactions in this session's store that share an entity key (hashed). Shared keys "
            "are context for an investigation, not evidence of fraud.",
        }
