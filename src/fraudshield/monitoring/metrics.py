"""Prometheus metrics (exposed at GET /metrics). Feature values are never used as labels."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        r = self.registry
        self.requests = Counter(
            "fraudshield_http_requests_total", "HTTP requests", ["route", "method", "status"], registry=r
        )
        self.request_latency = Histogram(
            "fraudshield_http_request_duration_seconds",
            "End-to-end request latency",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.predictions = Counter(
            "fraudshield_predictions_total",
            "Scored transactions",
            ["action", "source", "model_version"],
            registry=r,
        )
        self.inference_latency = Histogram(
            "fraudshield_inference_duration_seconds",
            "Model inference time (features + model + calibration)",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.score = Histogram(
            "fraudshield_fraud_probability",
            "Calibrated fraud probability",
            buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            registry=r,
        )
        self.errors = Counter("fraudshield_errors_total", "Errors by type", ["type"], registry=r)
        self.quality_warnings = Counter(
            "fraudshield_data_quality_warnings_total", "Data-quality warnings", registry=r
        )
        self.drift_psi = Gauge(
            "fraudshield_feature_drift_psi_max", "Max PSI across features (rolling window)", registry=r
        )
        self.score_psi = Gauge(
            "fraudshield_score_drift_psi", "PSI of score distribution (rolling window)", registry=r
        )
        self.replay_running = Gauge(
            "fraudshield_replay_running", "1 if transaction replay is running", registry=r
        )
        self.model_info = Gauge(
            "fraudshield_model_info", "Loaded model", ["model_version", "kind"], registry=r
        )
        self.reviews_last_hour = Gauge(
            "fraudshield_reviews_last_hour", "Manual reviews routed in the last hour", registry=r
        )
