"""Shared dependencies: application state accessors, API-key auth and rate limiting.

All dependencies are ``async def`` on purpose: they never block, and FastAPI would otherwise run
each sync dependency in the thread pool (several thread hops per request).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header, Request

from api.observability import TokenBucketLimiter, constant_time_equals
from fraudshield.config import Settings


class ApiError(Exception):
    """Error rendered with the standard error envelope; never includes internal details."""

    def __init__(
        self, status: int, code: str, message: str, details: Any = None, headers: dict[str, str] | None = None
    ):
        super().__init__(message)
        self.status, self.code, self.message, self.details, self.headers = (
            status,
            code,
            message,
            details,
            headers,
        )


@dataclass
class AppState:
    settings: Settings
    service: Any = None  # FraudService (None until a model is loaded)
    replay: Any = None
    profile: Any = None  # ServingProfile (set before the model loads, so /config always works)
    limiter: TokenBucketLimiter | None = None
    startup_error: str | None = None


async def get_state(request: Request) -> AppState:
    return request.app.state.fs


async def get_service(state: AppState = Depends(get_state)) -> Any:
    if state.service is None:
        raise ApiError(503, "model_unavailable", "No model is loaded; see GET /ready for details")
    return state.service


async def require_api_key(
    state: AppState = Depends(get_state), x_api_key: str | None = Header(default=None)
) -> None:
    expected = state.settings.api_key
    if expected is None:
        return  # auth disabled (local demo); documented in README / DEPLOYMENT
    if x_api_key is None or not constant_time_equals(x_api_key, expected.get_secret_value()):
        raise ApiError(401, "unauthorized", "missing or invalid X-API-Key header")


def check_rate(request: Request, state: AppState, cost: float = 1.0) -> None:
    if state.limiter is None:
        return
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    ok, retry = state.limiter.allow(key, cost)
    if not ok:
        raise ApiError(
            429,
            "rate_limited",
            "rate limit exceeded",
            headers={"Retry-After": f"{max(1, int(retry + 0.999))}"},
        )


def rate_limited(cost: float = 1.0) -> Any:
    async def dep(request: Request, state: AppState = Depends(get_state)) -> None:
        check_rate(request, state, cost)

    return dep
