"""Structured JSON logging, request IDs, timing, security headers and rate limiting."""

from __future__ import annotations

import json
import logging
import secrets
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("fraudshield.api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key in (
            "method",
            "path",
            "status",
            "duration_ms",
            "client",
            "count",
            "error_type",
            "model_version",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info and record.levelno >= logging.ERROR:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger("fraudshield")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


class RequestContextMiddleware:
    """Pure ASGI middleware (cheaper than ``BaseHTTPMiddleware`` and streaming-safe).

    Assigns/propagates ``X-Request-ID``, adds security headers, records Prometheus and live
    metrics, writes one structured log line per request and converts unhandled exceptions into
    the standard error envelope without leaking internals.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id", "")
        rid = incoming if 0 < len(incoming) <= 64 and incoming.isascii() else uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        state = scope["app"].state
        status = 500
        started = False
        path = scope.get("path", "")

        async def send_wrapper(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = rid
                for k, v in SECURITY_HEADERS.items():
                    headers.setdefault(k, v)
                if not path.startswith(("/docs", "/redoc", "/openapi.json")):
                    headers.setdefault("Content-Security-Policy", CSP)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:  # last line of defence: never leak internals to the client
            logger.error("unhandled error", exc_info=exc, extra={"error_type": type(exc).__name__})
            if hasattr(state, "metrics"):
                state.metrics.errors.labels("unhandled").inc()
            if hasattr(state, "monitor"):
                state.monitor.record_error(type(exc).__name__)
            if not started:
                body = {
                    "error": {
                        "code": "internal_error",
                        "request_id": rid,
                        "message": "an internal error occurred; quote the request_id when reporting it",
                    }
                }
                await JSONResponse(body, status_code=500)(scope, receive, send_wrapper)
        finally:
            ms = (time.perf_counter() - start) * 1000
            route_path = getattr(scope.get("route"), "path", "unmatched")
            method = scope.get("method", "")
            if hasattr(state, "metrics"):
                state.metrics.requests.labels(route_path, method, str(status)).inc()
                state.metrics.request_latency.labels(route_path).observe(ms / 1000)
            if (
                hasattr(state, "monitor")
                and route_path.startswith("/api/v1/transactions")
                and method == "POST"
            ):
                state.monitor.record_request(ms)
                if status >= 500:
                    state.monitor.record_error(f"http_{status}")
            if route_path not in ("/health", "/metrics", "/api/v1/stream", "unmatched"):
                logger.info(
                    "request",
                    extra={
                        "method": method,
                        "path": route_path,
                        "status": status,
                        "duration_ms": round(ms, 3),
                    },
                )
            request_id_var.reset(token)


class TokenBucketLimiter:
    """Per-client token bucket. In-memory: limits apply per API instance."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate, self.burst = rate, burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            if len(self._buckets) > 10_000:  # bound memory under many distinct clients
                self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < 60}
            return False, (cost - tokens) / self.rate


def constant_time_equals(a: str, b: str) -> bool:
    return secrets.compare_digest(a.encode(), b.encode())
