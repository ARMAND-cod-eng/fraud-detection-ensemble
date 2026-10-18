"""IEEE-CIS "v2" profile: authorisation-time fields + real-time velocity features.

The v1 IEEE-CIS benchmark used all 431 columns, including ~370 vendor-engineered ones
(C/D/V/id_) that a merchant would not compute itself. v2 restricts the model to a *field contract*
of raw values available when a card is authorised, and replaces the vendor aggregates with
our own velocity features from ``fraudshield.features.velocity``, which run identically offline
and online.

``IEEEPreprocessor`` has the same interface as ``fraudshield.features.engineering.FeatureBuilder``
(``transform``, ``fit``, ``feature_names``, ``profile``, ``percentile``, ``to_dict``), so the
existing artifact format, ``FraudModel``, TreeSHAP explanations and promotion gate apply unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.features.velocity import VELOCITY_SPECS

TIME = "TransactionDT"
AMOUNT = "TransactionAmt"
LABEL = "isFraud"

# Raw fields available at authorisation time (no vendor-engineered C/D/V/id_ aggregates).
NUMERIC_FIELDS: tuple[str, ...] = (
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
)
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
    "DeviceType",
    "DeviceInfo",
)
TX_CONTRACT_FIELDS: tuple[str, ...] = tuple(
    f for f in (*NUMERIC_FIELDS, *CATEGORICAL_FIELDS) if f not in ("DeviceType", "DeviceInfo")
)
IDENTITY_CONTRACT_FIELDS: tuple[str, ...] = ("DeviceType", "DeviceInfo")
CONTRACT_FIELDS: tuple[str, ...] = (*NUMERIC_FIELDS, *CATEGORICAL_FIELDS)
DERIVED_FEATURES: tuple[str, ...] = ("hour_of_day", "day_of_week", "amount_cents", "email_domains_match")
VELOCITY_FEATURES: tuple[str, ...] = tuple(s.name for s in VELOCITY_SPECS)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "TransactionAmt": "Transaction amount (USD)",
    "card1": "Card identifier field 1 (masked)",
    "card2": "Card identifier field 2 (masked)",
    "card3": "Card identifier field 3 (masked)",
    "card5": "Card identifier field 5 (masked)",
    "card4": "Card network",
    "card6": "Card type (debit / credit)",
    "addr1": "Billing region (masked)",
    "addr2": "Billing country (masked)",
    "dist1": "Distance between addresses (masked units)",
    "ProductCD": "Product code",
    "P_emaildomain": "Purchaser e-mail domain",
    "R_emaildomain": "Recipient e-mail domain",
    "DeviceType": "Device type (desktop / mobile)",
    "DeviceInfo": "Device model / operating system",
    **{f"M{i}": f"Match check M{i} (e.g. name or address match; meaning masked)" for i in range(1, 10)},
    "hour_of_day": "Hour of day (relative to the data's undisclosed reference time)",
    "day_of_week": "Day of week (relative)",
    "amount_cents": "Cents part of the amount (currency-conversion signal)",
    "email_domains_match": "Purchaser and recipient e-mail domains match",
}
_ENTITY_LABEL = {
    "card1": "card",
    "P_emaildomain": "e-mail domain",
    "DeviceInfo": "device model",
    "addr1": "billing region",
}
_FIELD_LABEL = {
    "card1": "cards",
    "P_emaildomain": "e-mail domains",
    "DeviceInfo": "device models",
    "addr1": "billing regions",
}
_WINDOW_LABEL = {3600: "1 h", 86_400: "24 h", 604_800: "7 days"}


def _describe(spec: Any) -> str:
    full_card = spec.entity[:4] == ("card1", "card2", "card3", "card5")
    rest = spec.entity[4:] if full_card else spec.entity[1:]
    entity = " and ".join(
        ["card" if full_card else _ENTITY_LABEL[spec.entity[0]], *(_ENTITY_LABEL[f] for f in rest)]
    )
    window = _WINDOW_LABEL.get(spec.window_s, f"{spec.window_s} s")
    return {
        "count": f"Earlier transactions with this {entity} in the last {window}",
        "sum_amount": f"Amount spent with this {entity} in the last {window}",
        "distinct": f"Distinct {_FIELD_LABEL.get(spec.field or '', spec.field)} seen with this {entity} in the last {window}",
        "secs_since_prev": f"Seconds since the previous transaction with this {entity} (within {window})",
        "amount_ratio_to_mean": f"Amount divided by this {entity}'s mean amount over the last {window}",
    }[spec.agg]


FEATURE_DESCRIPTIONS.update({s.name: _describe(s) for s in VELOCITY_SPECS})
REASON_TEMPLATES: dict[str, str] = {s.name: s.template for s in VELOCITY_SPECS}
# Masked identifiers: numeric codes whose order means nothing.
IDENTIFIER_FIELDS: frozenset[str] = frozenset({"card1", "card2", "card3", "card5", "addr1", "addr2"})
# Percentiles are shown to analysts only where "higher / lower than training" is meaningful.
NO_PERCENTILE: frozenset[str] = frozenset(
    {*CATEGORICAL_FIELDS, *IDENTIFIER_FIELDS, "email_domains_match", "day_of_week"}
)

_QUANTILE_LEVELS = np.linspace(0.0, 1.0, 201)


def _isna(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _num(v: Any) -> float:
    if _isna(v):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _col(data: Any, name: str) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        return data[name]
    v = data[name]
    return pd.Series(np.atleast_1d(np.asarray(v, dtype=object)))


class IEEEPreprocessor:
    """Contract fields (+ optional velocity features) -> fixed-order numeric matrix.

    Categorical fields are encoded with categories learned on the training period only; unseen or
    missing values become -1. Numeric NaNs stay NaN (XGBoost routes them natively).
    """

    version = "ieee-v2-1"
    kind = "ieee_v2"

    def __init__(self, velocity: bool = True) -> None:
        self.velocity = velocity
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = [*NUMERIC_FIELDS, *CATEGORICAL_FIELDS, *DERIVED_FEATURES]
        if velocity:
            self.feature_names += list(VELOCITY_FEATURES)
        self.profile: dict[str, dict[str, Any]] = {}

    @property
    def raw_columns(self) -> list[str]:
        return [TIME, *CONTRACT_FIELDS]

    def _encode(self, name: str, values: pd.Series) -> np.ndarray:
        cats = self.categories.get(name, [])
        s = values.astype(object)
        seen = s.where(s.isin(cats))  # unseen or missing -> NaN -> code -1
        return pd.Categorical(seen, categories=cats).codes.astype(np.float64)

    def transform_record(self, rec: Mapping[str, Any]) -> np.ndarray:
        """Single-transaction fast path (no pandas), numerically identical to ``transform``.

        ``tests/unit/test_ieee_v2.py`` checks the two paths agree row by row: the serving path must
        not reintroduce train/serve skew.
        """
        luts = self.__dict__.setdefault("_luts", {})
        out = np.empty(len(self.feature_names), dtype=np.float64)
        i = 0
        for f in NUMERIC_FIELDS:
            out[i] = _num(rec.get(f))
            i += 1
        for f in CATEGORICAL_FIELDS:
            lut = luts.get(f)
            if lut is None:
                lut = luts[f] = {c: k for k, c in enumerate(self.categories.get(f, []))}
            v = rec.get(f)
            out[i] = lut.get(v, -1) if isinstance(v, str) else -1
            i += 1
        t, amount = _num(rec.get(TIME)), out[0]
        p_mail, r_mail = rec.get("P_emaildomain"), rec.get("R_emaildomain")
        out[i] = (t / 3600.0) % 24.0
        out[i + 1] = np.floor(t / 86_400.0) % 7.0
        out[i + 2] = np.round(amount % 1.0, 3)
        out[i + 3] = np.nan if _isna(p_mail) or _isna(r_mail) else float(str(p_mail) == str(r_mail))
        i += 4
        if self.velocity:
            for f in VELOCITY_FEATURES:
                out[i] = _num(rec.get(f))
                i += 1
        return out[None, :]

    def transform(self, data: pd.DataFrame | Mapping[str, Sequence[Any] | np.ndarray]) -> np.ndarray:
        if not isinstance(data, pd.DataFrame):
            first = next(iter(data.values()), None)
            if first is None or np.ndim(first) == 0 or len(first) == 1:  # one transaction
                return self.transform_record({k: (v if np.ndim(v) == 0 else v[0]) for k, v in data.items()})
        cols: list[np.ndarray] = []
        for f in NUMERIC_FIELDS:
            cols.append(pd.to_numeric(_col(data, f), errors="coerce").to_numpy(dtype=np.float64))
        for f in CATEGORICAL_FIELDS:
            cols.append(self._encode(f, _col(data, f)))
        t = pd.to_numeric(_col(data, TIME), errors="coerce").to_numpy(dtype=np.float64)
        amount = cols[0]
        p_mail, r_mail = _col(data, "P_emaildomain"), _col(data, "R_emaildomain")
        match = np.where(
            p_mail.isna().to_numpy() | r_mail.isna().to_numpy(),
            np.nan,
            (p_mail.astype(str).to_numpy() == r_mail.astype(str).to_numpy()).astype(float),
        )
        cols += [
            (t / 3600.0) % 24.0,
            np.floor(t / 86_400.0) % 7.0,
            np.round(amount % 1.0, 3),
            match,
        ]
        if self.velocity:
            for f in VELOCITY_FEATURES:
                cols.append(pd.to_numeric(_col(data, f), errors="coerce").to_numpy(dtype=np.float64))
        return np.column_stack(cols)

    def fit(self, data: pd.DataFrame) -> IEEEPreprocessor:
        """Learn categories and the training profile (train period only)."""
        self.__dict__.pop("_luts", None)
        self.categories = {
            f: sorted({str(v) for v in data[f].dropna().unique()}) for f in CATEGORICAL_FIELDS if f in data
        }
        x = self.transform(data)
        self.profile = {}
        for j, name in enumerate(self.feature_names):
            col = x[:, j]
            col = col[~np.isnan(col)]
            if not len(col):
                continue
            self.profile[name] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "p001": float(np.quantile(col, 0.001)),
                "p999": float(np.quantile(col, 0.999)),
                "quantiles": [float(v) for v in np.quantile(col, _QUANTILE_LEVELS)],
            }
        return self

    def percentile(self, name: str, value: float) -> float | None:
        prof = self.profile.get(name)
        if name in NO_PERCENTILE or not prof or value is None or np.isnan(value):
            return None
        return float(np.interp(value, np.asarray(prof["quantiles"]), _QUANTILE_LEVELS * 100.0))

    def typical(self, name: str) -> float | None:
        prof = self.profile.get(name)
        return None if not prof else float(prof["quantiles"][100])  # training median

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "version": self.version,
            "velocity": self.velocity,
            "feature_names": self.feature_names,
            "categories": self.categories,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IEEEPreprocessor:
        p = cls(velocity=bool(payload["velocity"]))
        if list(payload["feature_names"]) != p.feature_names:
            raise ValueError("feature_names in artifact do not match this IEEEPreprocessor version")
        p.categories = {k: list(v) for k, v in payload.get("categories", {}).items()}
        p.profile = dict(payload.get("profile", {}))
        return p


def reason_text(feature: str, value: float, typical: float | None) -> str | None:
    """Plain-language sentence for a velocity feature, e.g. '7 earlier transactions on this card...'."""
    tpl = REASON_TEMPLATES.get(feature)
    if tpl is None or value is None or np.isnan(value):
        return None
    text = tpl.format(value=value)
    if tpl.startswith("{value:.0f} ") and round(value) == 1:  # "1 earlier transaction", not "transactions"
        for plural in ("transactions", "cards", "domains", "models", "regions"):
            if plural in text:
                text = text.replace(plural, plural[:-1], 1)
                break
    if typical is not None and not np.isnan(typical):
        text += f" (typical: {typical:,.0f})" if "ratio" not in feature else f" (typical: {typical:.1f}x)"
    return text
