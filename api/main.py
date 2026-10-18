"""FastAPI application factory.

Run locally:  uvicorn api.main:app --port 8000
The dashboard (dashboard/dist) is served from the same origin when it has been built.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.deps import ApiError, AppState
from api.observability import RequestContextMiddleware, TokenBucketLimiter, configure_logging, request_id_var
from api.routes import platform, transactions
from fraudshield import __version__
from fraudshield.config import Settings, get_settings
from fraudshield.decision_engine.policy import DecisionEngine, load_policy
from fraudshield.inference.scorer import FraudScorer
from fraudshield.monitoring.live import LiveMonitor
from fraudshield.monitoring.metrics import Metrics
from fraudshield.profiles import get_profile
from fraudshield.profiles.base import ServingProfile
from fraudshield.service import FraudService
from fraudshield.storage.store import PredictionStore
from fraudshield.streaming.broadcaster import Broadcaster
from fraudshield.streaming.replay import ReplayEngine

logger = logging.getLogger("fraudshield.api")


def _error(
    status: int, code: str, message: str, details: Any = None, headers: dict[str, str] | None = None
) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status, headers=headers)


def build_service(
    settings: Settings, metrics: Metrics, profile: ServingProfile | None = None
) -> FraudService:
    profile = profile or get_profile(settings)
    scorer = FraudScorer.from_registry(settings.artifacts_dir, settings.model_version, profile)
    policy = load_policy(settings.policy_path, scorer.review_threshold)
    engine = DecisionEngine(policy)
    store = PredictionStore(settings.db_path)
    store.add_policy_version(policy.version, "startup", policy.model_dump())
    monitor = LiveMonitor(scorer.reference, scorer.model.feature_builder.feature_names)
    return FraudService(
        scorer, engine, store, monitor, metrics, Broadcaster(), settings.max_stored_predictions
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(
            settings=settings,
            limiter=TokenBucketLimiter(settings.rate_limit_per_second, settings.rate_limit_burst),
        )
        app.state.fs = state
        app.state.metrics = metrics
        try:
            state.profile = get_profile(settings)
            service = build_service(settings, metrics, state.profile)
            online = state.profile.startup()  # e.g. warm the online feature store
            service.broadcaster.bind_loop(asyncio.get_running_loop())
            state.service = service
            app.state.monitor = service.monitor
            state.replay = ReplayEngine(service)
            logger.info(
                "model loaded",
                extra={"model_version": service.scorer.version, "profile": settings.profile, **online},
            )
        except Exception as exc:  # keep serving /health and /ready so the failure is observable
            state.startup_error = f"{type(exc).__name__}: {exc}"
            logger.error("startup failed: %s", state.startup_error)
        yield
        if state.replay is not None:
            await state.replay.stop()
        if state.service is not None:
            state.service.store.close()

    app = FastAPI(
        title="FraudShield AI",
        version=__version__,
        description="Real-time fraud scoring, decisioning, investigation and monitoring API.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
        ][:20]
        return _error(422, "validation_error", "request body or parameters failed validation", details)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc, extra={"error_type": type(exc).__name__})
        if hasattr(app.state, "monitor"):
            app.state.monitor.record_error(type(exc).__name__)
        metrics.errors.labels("unhandled").inc()
        return _error(
            500, "internal_error", "an internal error occurred; quote the request_id when reporting it"
        )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request) -> JSONResponse:
        """Readiness: a verified model is loaded, the store responds and a policy is active."""
        state: AppState = request.app.state.fs
        svc = state.service
        checks = {
            "model_loaded": svc is not None,
            "store": bool(svc and svc.store.ping()),
            "policy": bool(svc and svc.engine.policy),
        }
        ok = all(checks.values())
        body: dict[str, Any] = {"status": "ready" if ok else "not_ready", "checks": checks}
        if svc is not None:
            body["model_version"] = svc.scorer.version
        if state.startup_error:
            body["reason"] = state.startup_error.split(":")[0]  # error class only; details are in the logs
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    def prometheus_metrics(request: Request) -> PlainTextResponse:
        state: AppState = request.app.state.fs
        if state.service is not None:
            drift = state.service.monitor.drift()
            if drift.get("available") and drift.get("reliable"):
                metrics.drift_psi.set(drift.get("max_feature_psi", 0.0))
                if "score" in drift:
                    metrics.score_psi.set(drift["score"]["psi"])
        return PlainTextResponse(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.include_router(transactions.build_scoring_router(settings.profile))
    app.include_router(transactions.router)
    app.include_router(platform.router)

    dist = settings.dashboard_dir
    if dist is not None and (Path(dist) / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=Path(dist) / "assets"), name="assets")
        index = Path(dist) / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (Path(dist) / path).resolve()
            if path and candidate.is_file() and Path(dist).resolve() in candidate.parents:
                return FileResponse(candidate)
            if path.startswith("api/"):
                raise ApiError(404, "not_found", "unknown API route")
            return FileResponse(index)

    return app


app = create_app()
