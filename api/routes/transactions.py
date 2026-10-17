"""Scoring, transaction listing, explanations and analyst review."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from api.deps import ApiError, AppState, check_rate, get_service, get_state, rate_limited, require_api_key
from api.schemas import TXN_ID_PATTERN, BatchIn, BatchOut, ErrorOut, ReviewIn, ScoreOut, TransactionIn
from fraudshield.service import ExplanationUnavailable, IdempotencyConflict, ScoreRequest

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])
ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut},
    409: {"model": ErrorOut},
    422: {"model": ErrorOut},
    429: {"model": ErrorOut},
    503: {"model": ErrorOut},
}


async def _score(state: AppState, service: Any, txns: list[TransactionIn]) -> list[dict[str, Any]]:
    reqs = [ScoreRequest(features=t.to_record(), transaction_id=t.transaction_id) for t in txns]
    timeout = state.settings.inference_timeout_ms / 1000
    try:
        return await asyncio.wait_for(run_in_threadpool(service.score, reqs, "api"), timeout=timeout)
    except TimeoutError:
        service.monitor.record_error("inference_timeout")
        service.metrics.errors.labels("inference_timeout").inc()
        raise ApiError(
            503,
            "inference_timeout",
            f"scoring exceeded {state.settings.inference_timeout_ms} ms; retry with the same transaction_id "
            "(idempotent)",
            headers={"Retry-After": "1"},
        ) from None
    except IdempotencyConflict as exc:
        service.metrics.errors.labels("idempotency_conflict").inc()
        raise ApiError(
            409,
            "idempotency_conflict",
            f"transaction_id {exc.transaction_id!r} was already scored with a different payload",
        ) from None


@router.post(
    "/score",
    response_model=ScoreOut,
    responses=ERRORS,
    dependencies=[Depends(require_api_key), Depends(rate_limited())],
)
async def score_transaction(
    txn: TransactionIn, state: AppState = Depends(get_state), service: Any = Depends(get_service)
) -> dict[str, Any]:
    """Score one transaction: calibrated fraud probability + policy decision."""
    return (await _score(state, service, [txn]))[0]


@router.post("/batch", response_model=BatchOut, responses=ERRORS, dependencies=[Depends(require_api_key)])
async def score_batch(
    batch: BatchIn,
    request: Request,
    state: AppState = Depends(get_state),
    service: Any = Depends(get_service),
) -> dict[str, Any]:
    """Score up to ``FRAUDSHIELD_MAX_BATCH_SIZE`` transactions in one vectorised call."""
    n = len(batch.transactions)
    if n > state.settings.max_batch_size:
        raise ApiError(422, "batch_too_large", f"batch of {n} exceeds limit {state.settings.max_batch_size}")
    check_rate(request, state, cost=float(n))  # a batch consumes one rate-limit token per transaction
    results = await _score(state, service, batch.transactions)
    return {"results": results, "count": len(results), "model_version": service.scorer.version}


@router.get("")
def list_transactions(
    service: Any = Depends(get_service),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    action: Literal["approve", "review", "decline"] | None = None,
    risk_level: Literal["low", "medium", "high"] | None = None,
    review_status: Literal["none", "open", "confirmed_fraud", "legitimate", "escalated"] | None = None,
    alerts_only: bool = False,
    search: str | None = Query(None, max_length=64, pattern=r"^[A-Za-z0-9_\-:.]*$"),
    min_probability: float | None = Query(None, ge=0, le=1),
) -> dict[str, Any]:
    items, total = service.store.list_predictions(
        limit, offset, action, risk_level, review_status, alerts_only, search, min_probability
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{transaction_id}", responses={404: {"model": ErrorOut}})
def get_transaction(transaction_id: str, service: Any = Depends(get_service)) -> dict[str, Any]:
    if not re.fullmatch(TXN_ID_PATTERN, transaction_id):
        raise ApiError(422, "invalid_transaction_id", "transaction_id has an invalid format")
    row = service.store.get(transaction_id)
    if row is None:
        raise ApiError(404, "not_found", f"transaction {transaction_id!r} not found")
    row.pop("payload_hash", None)
    gt = row.pop("ground_truth", None)
    # Ground truth is only revealed once an analyst has closed the case (it simulates a label
    # that arrives later), so the investigation view cannot leak the answer.
    row["ground_truth_revealed"] = gt if row["review_status"] in ("confirmed_fraud", "legitimate") else None
    row["history"] = service.store.case_events(transaction_id)
    return row


@router.get("/{transaction_id}/explanation", responses={404: {"model": ErrorOut}, 409: {"model": ErrorOut}})
def get_explanation(transaction_id: str, service: Any = Depends(get_service)) -> dict[str, Any]:
    try:
        exp = service.explanation(transaction_id)
    except ExplanationUnavailable as exc:
        raise ApiError(409, "explanation_unavailable", str(exc)) from None
    if exp is None:
        raise ApiError(404, "not_found", f"transaction {transaction_id!r} not found")
    return exp


@router.post(
    "/{transaction_id}/review", dependencies=[Depends(require_api_key)], responses={404: {"model": ErrorOut}}
)
def review_transaction(
    transaction_id: str, body: ReviewIn, service: Any = Depends(get_service)
) -> dict[str, Any]:
    """Record an analyst decision or note (persisted with an audit trail)."""
    if service.store.get(transaction_id) is None:
        raise ApiError(404, "not_found", f"transaction {transaction_id!r} not found")
    event = "note" if body.decision == "note" else "decision"
    decision = None if body.decision == "note" else body.decision
    return service.store.add_case_event(transaction_id, body.analyst, event, decision, body.notes)
