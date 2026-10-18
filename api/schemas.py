"""Pydantic request/response contracts for the public API."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

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


_TEXT = r"^[ -~]{1,64}$"  # printable ASCII (domains, device strings like "SM-G892A Build/NRD90M")


def _num(desc: str, hi: float = 1e7) -> Any:
    return Field(None, ge=0, le=hi, allow_inf_nan=False, description=desc)


def _text(desc: str, max_length: int = 64) -> Any:
    return Field(None, pattern=_TEXT, max_length=max_length, description=desc)


class IEEETransactionIn(BaseModel):
    """IEEE-CIS authorisation-time field contract (``FRAUDSHIELD_PROFILE=ieee``).

    Only raw values a merchant has when the card is authorised; the vendor-engineered C/D/V/id_
    aggregates are not part of the contract. Velocity features are computed server-side from the
    online feature store. Missing fields are allowed (the model was trained with missing values).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": "ieee_000123",
                "amount": 117.0,
                "time_seconds": 13_000_000,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 361.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "debit",
                "addr1": 315.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "M4": "M0",
                "DeviceType": "desktop",
                "DeviceInfo": "Windows",
            }
        },
    )

    transaction_id: str | None = Field(
        None,
        pattern=TXN_ID_PATTERN,
        description="Client-supplied ID; doubles as an idempotency key. Generated when omitted.",
    )
    amount: float = Field(..., ge=0, le=1_000_000, allow_inf_nan=False, description="TransactionAmt (USD)")
    time_seconds: float = Field(
        ...,
        ge=0,
        le=1e9,
        allow_inf_nan=False,
        description="TransactionDT: seconds since the dataset's undisclosed reference time. Events must "
        "arrive in time order; earlier events are processed at the stream watermark and flagged.",
    )
    ProductCD: str | None = _text("Product code", 8)
    card1: float | None = _num("Card identifier field 1 (masked)")
    card2: float | None = _num("Card identifier field 2 (masked)")
    card3: float | None = _num("Card identifier field 3 (masked)")
    card4: str | None = _text("Card network")
    card5: float | None = _num("Card identifier field 5 (masked)")
    card6: str | None = _text("Card type")
    addr1: float | None = _num("Billing region (masked)")
    addr2: float | None = _num("Billing country (masked)")
    dist1: float | None = _num("Distance between addresses (masked units)")
    P_emaildomain: str | None = _text("Purchaser e-mail domain")
    R_emaildomain: str | None = _text("Recipient e-mail domain")
    M1: str | None = _text("Match check M1", 8)
    M2: str | None = _text("Match check M2", 8)
    M3: str | None = _text("Match check M3", 8)
    M4: str | None = _text("Match check M4", 8)
    M5: str | None = _text("Match check M5", 8)
    M6: str | None = _text("Match check M6", 8)
    M7: str | None = _text("Match check M7", 8)
    M8: str | None = _text("Match check M8", 8)
    M9: str | None = _text("Match check M9", 8)
    DeviceType: str | None = _text("Device type (desktop / mobile)", 16)
    DeviceInfo: str | None = _text("Device model / operating system")

    def to_record(self) -> dict[str, Any]:
        rec = self.model_dump(exclude={"transaction_id", "amount", "time_seconds"})
        rec["TransactionDT"] = self.time_seconds
        rec["TransactionAmt"] = self.amount
        return rec


T = TypeVar("T", TransactionIn, IEEETransactionIn)


class Batch(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")
    transactions: list[T] = Field(..., min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _unique_ids(self) -> Batch[T]:
        ids = [t.transaction_id for t in self.transactions if t.transaction_id]
        if len(ids) != len(set(ids)):
            raise ValueError("transaction_id values must be unique within a batch")
        return self


BatchIn = Batch[TransactionIn]
REQUEST_MODELS: dict[str, type[TransactionIn] | type[IEEETransactionIn]] = {
    "creditcard": TransactionIn,
    "ieee": IEEETransactionIn,
}


class ScoreOut(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., description="Calibrated probability of fraud (Platt scaling)")
    raw_model_score: float = Field(..., description="Uncalibrated model output, for diagnostics")
    risk_level: Literal["low", "medium", "high"]
    recommended_action: Literal["approve", "review", "decline"]
    decision_reasons: list[str]
    top_risk_factors: list[str] = Field(
        ...,
        description="For alerts: the strongest risk-increasing attributions as plain sentences "
        "(model associations, not evidence). Empty for approvals.",
    )
    expected_loss_eur: float = Field(
        ...,
        description="probability x amount x loss_rate (policy assumption), in the dataset's currency "
        "(EUR for creditcard, USD for ieee; see GET /api/v1/config)",
    )
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
