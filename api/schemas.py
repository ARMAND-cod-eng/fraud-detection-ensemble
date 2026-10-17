"""Pydantic request/response contracts for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from fraudshield.data.schema import V_COLUMNS

TXN_ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,64}$"

PCAFeatures = create_model(  # type: ignore[call-overload]
    "PCAFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{v: (float, Field(..., allow_inf_nan=False, ge=-1e4, le=1e4)) for v in V_COLUMNS},
)
PCAFeatures.__doc__ = "The 28 anonymised PCA components (V1-V28) published with the dataset."


class TransactionIn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": "txn_000123",
                "amount": 149.62,
                "time_seconds": 43_200,
                "features": {v: 0.0 for v in V_COLUMNS},
            }
        },
    )

    transaction_id: str | None = Field(
        None,
        pattern=TXN_ID_PATTERN,
        description="Client-supplied ID; doubles as an idempotency key. Generated when omitted.",
    )
    amount: float = Field(
        ..., ge=0, le=1_000_000, allow_inf_nan=False, description="Transaction amount (EUR)"
    )
    time_seconds: float = Field(
        ...,
        ge=0,
        le=1e9,
        allow_inf_nan=False,
        description="Seconds since the reference start of the capture window (dataset 'Time' column)",
    )
    features: PCAFeatures  # type: ignore[valid-type]

    def to_record(self) -> dict[str, float]:
        rec: dict[str, float] = self.features.model_dump()  # type: ignore[attr-defined]
        rec["Time"] = self.time_seconds
        rec["Amount"] = self.amount
        return rec


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionIn] = Field(..., min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _unique_ids(self) -> BatchIn:
        ids = [t.transaction_id for t in self.transactions if t.transaction_id]
        if len(ids) != len(set(ids)):
            raise ValueError("transaction_id values must be unique within a batch")
        return self


class ScoreOut(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., description="Calibrated probability of fraud (Platt scaling)")
    raw_model_score: float = Field(..., description="Uncalibrated model output, for diagnostics")
    risk_level: Literal["low", "medium", "high"]
    recommended_action: Literal["approve", "review", "decline"]
    decision_reasons: list[str]
    expected_loss_eur: float = Field(..., description="probability x amount x loss_rate (policy assumption)")
    review_capacity_exceeded: bool
    model_version: str
    policy_version: str
    scored_at: str
    inference_ms: float
    explanation_available: bool
    explanation_url: str
    data_quality_warnings: list[str]
    synthetic: bool
    idempotent_replay: bool = Field(
        ..., description="True when this ID was already scored with the same payload"
    )
    processing_ms: float


class BatchOut(BaseModel):
    results: list[ScoreOut]
    count: int
    model_version: str


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["confirmed_fraud", "legitimate", "escalated", "note"]
    notes: str | None = Field(None, max_length=2000)
    analyst: str = Field("demo-analyst", pattern=r"^[A-Za-z0-9_\-. ]{1,64}$")


class ReplayStartIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Literal["historical", "enriched", "synthetic_drift"] = "enriched"
    rate_tps: float = Field(20.0, gt=0, le=1000)
    count: int = Field(2000, ge=1, le=200_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class ReplayRateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rate_tps: float = Field(..., gt=0, le=1000)


class PolicyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_threshold: float = Field(..., gt=0, lt=1)
    decline_threshold: float = Field(..., gt=0, le=1)
    expected_loss_review: bool = False
    max_reviews_per_hour: int = Field(120, ge=1, le=100_000)
    review_cost: float | None = Field(None, ge=0, le=10_000)
    false_decline_cost: float | None = Field(None, ge=0, le=10_000)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorOut(BaseModel):
    error: ErrorBody
