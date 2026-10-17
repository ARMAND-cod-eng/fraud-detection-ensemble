"""Model info, policy, overview/monitoring aggregates, replay control and the live event stream."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.deps import ApiError, AppState, get_service, get_state, require_api_key
from api.schemas import PolicyIn, ReplayRateIn, ReplayStartIn
from fraudshield.data.schema import AMOUNT_COLUMN, LABEL_COLUMN, TIME_COLUMN
from fraudshield.decision_engine.policy import CostAssumptions, PolicyConfig
from fraudshield.decision_engine.simulate import capacity_curve, simulate_policy
from fraudshield.models.artifact import ModelRegistry
from fraudshield.streaming.replay import SCENARIOS

router = APIRouter(prefix="/api/v1")
_sim_cache: dict[str, Any] = {}
_sim_lock = threading.Lock()


# --------------------------------------------------------------------------- models
@router.get("/models/info", tags=["models"])
def model_info(service: Any = Depends(get_service)) -> dict[str, Any]:
    return service.scorer.info()


@router.get("/models/evaluation", tags=["models"])
def model_evaluation(service: Any = Depends(get_service)) -> dict[str, Any]:
    """Offline evaluation report stored with the artifact (curves, CIs, costs, latency)."""
    ev = service.scorer.evaluation()
    if ev is None:
        raise ApiError(404, "not_found", "this model artifact has no evaluation report")
    ev.pop("config", None)
    return ev


@router.get("/models/versions", tags=["models"])
def model_versions(state: AppState = Depends(get_state)) -> dict[str, Any]:
    reg = ModelRegistry(state.settings.artifacts_dir)
    return {
        "production": reg.production(),
        "versions": reg.versions(),
        "history": reg.history(),
        "serving": state.service.scorer.version if state.service else None,
    }


# --------------------------------------------------------------------------- policy
@router.get("/policy", tags=["policy"])
def get_policy(service: Any = Depends(get_service)) -> dict[str, Any]:
    p = service.engine.policy
    return {
        "version": p.version,
        "config": p.model_dump(),
        "model_review_threshold": service.scorer.review_threshold,
        "reviews_last_hour": service.engine.reviews_last_hour(),
        "history": [
            {"version": h["version"], "at": h["at"], "actor": h.get("actor", "startup")}
            for h in service.engine.history[-20:]
        ],
        "disclaimer": "Example policy with illustrative cost assumptions; not a validated financial decision policy.",
    }


def _policy_from(body: PolicyIn, base: PolicyConfig) -> PolicyConfig:
    costs = base.cost_assumptions.model_dump()
    if body.review_cost is not None:
        costs["review_cost"] = body.review_cost
    if body.false_decline_cost is not None:
        costs["false_decline_cost"] = body.false_decline_cost
    try:
        return PolicyConfig(
            policy_id=base.policy_id if base.policy_id.endswith("-custom") else f"{base.policy_id}-custom",
            description="Adjusted from the dashboard",
            review_threshold=body.review_threshold,
            decline_threshold=body.decline_threshold,
            expected_loss_review=body.expected_loss_review,
            max_reviews_per_hour=body.max_reviews_per_hour,
            cost_assumptions=CostAssumptions(**costs),
        )
    except ValueError as exc:
        raise ApiError(
            422, "invalid_policy", str(exc).splitlines()[-1] if str(exc) else "invalid policy"
        ) from None


@router.put("/policy", tags=["policy"], dependencies=[Depends(require_api_key)])
def update_policy(body: PolicyIn, service: Any = Depends(get_service)) -> dict[str, Any]:
    """Replace the active policy. Every later decision records the new policy version."""
    policy = _policy_from(body, service.engine.policy)
    service.engine.update(policy, actor="api")
    service.store.add_policy_version(policy.version, "api", policy.model_dump())
    return {"version": policy.version, "config": policy.model_dump()}


def _held_out_scores(state: AppState, service: Any) -> dict[str, Any]:
    key = service.scorer.version
    with _sim_lock:
        if key not in _sim_cache:
            path = state.settings.replay_file
            if not path.exists():
                raise ApiError(404, "not_found", "held-out replay file not found; run training first")
            df = pd.read_csv(path)
            cols = {c: df[c].to_numpy(dtype=np.float64) for c in df.columns if c != LABEL_COLUMN}
            p = service.scorer.score_columns(cols).probability
            hours = float(df[TIME_COLUMN].max() - df[TIME_COLUMN].min()) / 3600
            _sim_cache.clear()
            _sim_cache[key] = {
                "y": df[LABEL_COLUMN].to_numpy(),
                "p": p,
                "amount": df[AMOUNT_COLUMN].to_numpy(),
                "hours": hours,
            }
        return _sim_cache[key]


@router.post("/policy/simulate", tags=["policy"])
def simulate(
    body: PolicyIn, state: AppState = Depends(get_state), service: Any = Depends(get_service)
) -> dict[str, Any]:
    """What-if: apply a policy to the labelled held-out test window and cost it."""
    policy = _policy_from(body, service.engine.policy)
    d = _held_out_scores(state, service)
    caps = sorted({10, 25, 50, 100, 200, body.max_reviews_per_hour})
    return {
        "data": "held-out test window (labelled); the default thresholds were chosen on validation, not here",
        "window_hours": d["hours"],
        "result": simulate_policy(policy, d["y"], d["p"], d["amount"]),
        "capacity": capacity_curve(
            d["y"], d["p"], d["amount"], d["hours"], caps, policy.cost_assumptions.loss_rate
        ),
        "assumptions": policy.cost_assumptions.model_dump(),
    }


# --------------------------------------------------------------------------- aggregates
@router.get("/overview", tags=["monitoring"])
def overview(state: AppState = Depends(get_state), service: Any = Depends(get_service)) -> dict[str, Any]:
    stats = service.store.overview()
    snap = service.monitor.snapshot()
    return {
        "stats": stats,
        "latency": {
            "request_ms": snap["request_latency_ms"],
            "inference_ms": snap["inference_latency_ms_per_txn"],
        },
        "throughput_tps_last_60s": snap["throughput_tps_last_60s"],
        "activity": service.store.timeseries(bucket_s=10, points=60),
        "model": {
            "version": service.scorer.version,
            "kind": service.scorer.model.kind,
            "status": "serving",
            "loaded_at": service.scorer.loaded_at,
        },
        "policy_version": service.engine.policy.version,
        "replay": state.replay.status.to_dict() if state.replay else None,
        "exposure_note": "Expected fraud amount = Σ calibrated probability × amount. A model estimate, not confirmed losses.",
    }


@router.get("/monitoring", tags=["monitoring"])
def monitoring(service: Any = Depends(get_service)) -> dict[str, Any]:
    snap = service.monitor.snapshot()
    labelled = service.store.labelled_outcomes()
    perf: dict[str, Any] = {
        "available": False,
        "n": len(labelled),
        "note": "Ground truth exists only for replayed historical rows (simulated delayed labels).",
    }
    if labelled:
        p = np.array([x[0] for x in labelled])
        y = np.array([x[1] for x in labelled])
        flagged = np.array([x[2] != "approve" for x in labelled])
        tp = int((flagged & (y == 1)).sum())
        perf.update(
            {
                "available": True,
                "frauds": int(y.sum()),
                "alerts": int(flagged.sum()),
                "true_positives": tp,
                "precision": tp / flagged.sum() if flagged.sum() else None,
                "recall": tp / y.sum() if y.sum() else None,
                "mean_predicted": float(p.mean()),
                "observed_rate": float(y.mean()),
            }
        )
    return {
        **snap,
        "drift": service.monitor.drift(),
        "throughput_series": service.monitor.throughput_series(120),
        "labelled_performance": perf,
        "analyst_feedback": service.store.analyst_feedback_outcomes(),
        "stream_subscribers": service.broadcaster.subscriber_count,
    }


# --------------------------------------------------------------------------- replay
def _replay(state: AppState) -> Any:
    if state.replay is None:
        raise ApiError(503, "replay_unavailable", "replay engine is not initialised (no model loaded)")
    return state.replay


@router.get("/replay/scenarios", tags=["replay"])
def replay_scenarios() -> dict[str, str]:
    return SCENARIOS


@router.get("/replay/status", tags=["replay"])
def replay_status(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _replay(state).status.to_dict()


@router.post("/replay/start", tags=["replay"], dependencies=[Depends(require_api_key)])
async def replay_start(body: ReplayStartIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    rep = _replay(state)
    try:
        status = await rep.start(body.scenario, body.rate_tps, body.count, body.seed)
    except RuntimeError as exc:
        raise ApiError(409, "replay_running", str(exc)) from None
    return status.to_dict()


@router.post("/replay/pause", tags=["replay"], dependencies=[Depends(require_api_key)])
def replay_pause(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _replay(state).pause().to_dict()


@router.post("/replay/resume", tags=["replay"], dependencies=[Depends(require_api_key)])
def replay_resume(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _replay(state).resume().to_dict()


@router.post("/replay/stop", tags=["replay"], dependencies=[Depends(require_api_key)])
async def replay_stop(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return (await _replay(state).stop()).to_dict()


@router.patch("/replay", tags=["replay"], dependencies=[Depends(require_api_key)])
def replay_rate(body: ReplayRateIn, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return _replay(state).set_rate(body.rate_tps).to_dict()


@router.post("/admin/reset", tags=["admin"], dependencies=[Depends(require_api_key)])
async def reset_demo(
    state: AppState = Depends(get_state), service: Any = Depends(get_service)
) -> dict[str, str]:
    """Clear stored predictions and case history (demo convenience)."""
    if state.replay and state.replay.running:
        await state.replay.stop()
    service.store.reset()
    return {"status": "reset"}


# --------------------------------------------------------------------------- stream
@router.get("/stream", tags=["monitoring"])
async def stream(request: Request, service: Any = Depends(get_service)) -> StreamingResponse:
    """Server-sent events: one ``transaction`` event per scored transaction, ``stats`` every 2 s."""
    sub = service.broadcaster.subscribe()
    state: AppState = request.app.state.fs

    async def gen() -> AsyncIterator[str]:
        last_stats = 0.0
        try:
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev['data'])}\n\n"
                except TimeoutError:
                    yield ": keep-alive\n\n"
                now = time.time()
                if now - last_stats >= 2.0:
                    last_stats = now
                    stats = {
                        "dropped": sub.dropped,
                        "replay": state.replay.status.to_dict() if state.replay else None,
                    }
                    yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
        finally:
            service.broadcaster.unsubscribe(sub)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
