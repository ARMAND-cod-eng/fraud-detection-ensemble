"""In-process rolling statistics for the monitoring dashboard.

Ground-truth labels for fraud arrive late (chargebacks take weeks), so live monitoring here
tracks what is observable immediately: latency, throughput, errors, input drift, score drift
and alert rate. Model accuracy is shown only where labels genuinely exist (replayed historical
data or analyst decisions) and is labelled as such.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

import numpy as np

from fraudshield.monitoring.drift import drift_report


class LiveMonitor:
    def __init__(
        self, reference: dict[str, Any] | None, feature_names: list[str], window: int = 5000
    ) -> None:
        self.reference = reference
        self.feature_names = feature_names
        self._lock = threading.Lock()
        self._features: deque[np.ndarray] = deque(maxlen=window)
        self._scores: deque[float] = deque(maxlen=window)
        self._inference_ms: deque[float] = deque(maxlen=window)
        self._request_ms: deque[tuple[float, float]] = deque(maxlen=window)  # (ts, ms)
        self._events: deque[tuple[float, int, int]] = deque(maxlen=200_000)  # (ts, n, alerts)
        self._errors: deque[tuple[float, str]] = deque(maxlen=10_000)
        self._quality: Counter[str] = Counter()
        self.started_at = time.time()
        self._drift_cache: tuple[float, dict[str, Any]] | None = None

    def record_scores(self, base: np.ndarray, probs: np.ndarray, alerts: int, inference_ms: float) -> None:
        now = time.time()
        with self._lock:
            for row, p in zip(base, probs, strict=True):
                self._features.append(row)
                self._scores.append(float(p))
            self._inference_ms.append(inference_ms / max(1, len(probs)))
            self._events.append((now, len(probs), alerts))

    def record_request(self, ms: float) -> None:
        with self._lock:
            self._request_ms.append((time.time(), ms))

    def record_error(self, kind: str) -> None:
        with self._lock:
            self._errors.append((time.time(), kind))

    def record_quality(self, warnings: list[str]) -> None:
        if warnings:
            with self._lock:
                for w in warnings:
                    self._quality[w.split("=")[0].split(" ")[0]] += 1

    @staticmethod
    def _pct(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
        a = np.asarray(values)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()),
            "n": len(a),
        }

    def drift(self, max_age_s: float = 5.0) -> dict[str, Any]:
        now = time.time()
        if self._drift_cache and now - self._drift_cache[0] < max_age_s:
            return self._drift_cache[1]
        with self._lock:
            feats = np.vstack(self._features) if self._features else np.empty((0, len(self.feature_names)))
            scores = np.asarray(self._scores)
        if self.reference is None:
            rep: dict[str, Any] = {"available": False, "reason": "model artifact has no reference profile"}
        else:
            rep = {"available": True, **drift_report(self.reference, feats, self.feature_names, scores)}
            exp = self.reference.get("expected_alert_rate")
            rep["expected_alert_rate"] = exp
        self._drift_cache = (now, rep)
        return rep

    def throughput_series(self, seconds: int = 120) -> list[dict[str, float]]:
        now = int(time.time())
        buckets = {t: [0, 0] for t in range(now - seconds + 1, now + 1)}
        with self._lock:
            for ts, n, a in self._events:
                t = int(ts)
                if t in buckets:
                    buckets[t][0] += n
                    buckets[t][1] += a
        return [{"t": t, "transactions": v[0], "alerts": v[1]} for t, v in sorted(buckets.items())]

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            req = [ms for ts, ms in self._request_ms]
            req_recent = [ms for ts, ms in self._request_ms if ts >= now - 60]
            inf = list(self._inference_ms)
            errors_5m = Counter(kind for ts, kind in self._errors if ts >= now - 300)
            errors_total = Counter(kind for _, kind in self._errors)
            last_60 = sum(n for ts, n, _ in self._events if ts >= now - 60)
            alerts_window = sum(a for ts, _, a in self._events if ts >= now - 300)
            n_window = sum(n for ts, n, _ in self._events if ts >= now - 300)
            quality = dict(self._quality.most_common(10))
        return {
            "uptime_s": now - self.started_at,
            "request_latency_ms": self._pct(req),
            "request_latency_ms_last_60s": self._pct(req_recent),
            "inference_latency_ms_per_txn": self._pct(inf),
            "throughput_tps_last_60s": last_60 / 60.0,
            "alert_rate_last_5m": alerts_window / n_window if n_window else None,
            "errors_last_5m": dict(errors_5m),
            "errors_total": dict(errors_total),
            "data_quality_warnings": quality,
        }
