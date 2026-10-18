"""Serving profile interface: everything on the online path that depends on the dataset.

The scoring service, replay engine and API are dataset-agnostic. A profile supplies the raw field
contract, the request -> model-input pipeline (including online features), data-quality checks,
entity keys for linked activity, replay sources and analyst-facing text.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

Record = dict[str, Any]
ReplayItem = tuple[Record, int | None, bool]  # (raw record, delayed ground truth, synthetic?)


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def json_value(v: Any) -> Any:
    """JSON-safe scalar: NaN/None -> None, numpy scalars -> Python, strings unchanged."""
    if isinstance(v, np.generic):
        v = v.item()
    if is_missing(v):
        return None
    if isinstance(v, bool | int | float):
        return float(v)
    return str(v)


def generic_reason(description: str, display: str, percentile: float | None) -> str:
    if percentile is None:
        return f"{description}: {display}"
    side = f"higher than {percentile:.0f}%" if percentile >= 50 else f"lower than {100 - percentile:.0f}%"
    return f"{description}: {display} ({side} of training transactions)"


class ServingProfile:
    name: str = ""
    preprocessor_kind: str = ""  # ``kind`` of the artifact's feature builder this profile serves
    raw_fields: tuple[str, ...] = ()
    time_field: str = ""
    amount_field: str = ""
    label_field: str = ""
    currency: str = "EUR"
    data_attribution: str = ""
    scenarios: ClassVar[dict[str, str]] = {}
    ordered_stream: bool = False  # replay must preserve event order (stateful online features)
    linked_activity: bool = False

    # ------------------------------------------------------------------ lifecycle
    def check_model(self, model: Any) -> None:
        kind = getattr(model.feature_builder, "kind", "creditcard")
        if kind != self.preprocessor_kind:
            raise RuntimeError(
                f"profile {self.name!r} cannot serve a model whose feature builder is {kind!r}; "
                "check FRAUDSHIELD_PROFILE and FRAUDSHIELD_ARTIFACTS_DIR"
            )

    def startup(self) -> dict[str, Any]:
        """Prepare online state (e.g. warm the feature store). Returns facts for logs/readiness."""
        return {}

    def status(self) -> dict[str, Any]:
        return {"profile": self.name}

    # ------------------------------------------------------------------ request path
    def payload_hash(self, record: Mapping[str, Any]) -> str:
        canonical = json.dumps([repr(json_value(record.get(f))) for f in self.raw_fields])
        return hashlib.sha256(canonical.encode()).hexdigest()

    def prepare(self, records: list[Record]) -> tuple[list[Record], list[list[str]]]:
        """Online enrichment for *new* transactions (may update state). Returns (records, warnings)."""
        return records, [[] for _ in records]

    def base_features(self, model: Any, records: list[Record]) -> np.ndarray:
        cols = {
            c: np.fromiter((float(r[c]) for r in records), dtype=np.float64, count=len(records))
            for c in self.raw_fields
        }
        return np.asarray(model.base_features(cols))

    def stored_features(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {f: json_value(record.get(f)) for f in self.raw_fields}

    def warm_up_record(self) -> Record:
        return {c: 0.0 for c in self.raw_fields}

    def quality_warnings(self, model: Any, base_row: np.ndarray, record: Mapping[str, Any]) -> list[str]:
        warnings: list[str] = []
        prof = model.feature_builder.profile
        for j, name in enumerate(model.feature_builder.feature_names):
            p, v = prof.get(name), base_row[j]
            if p and not np.isnan(v) and not (p["min"] <= v <= p["max"]):
                warnings.append(f"{name}={v:.4g} outside training range [{p['min']:.4g}, {p['max']:.4g}]")
        return warnings

    # ------------------------------------------------------------------ analyst context
    def feature_description(self, feature: str) -> str:
        return feature

    def display_value(self, feature: str, value: float, record: Mapping[str, Any] | None) -> str:
        return "missing" if np.isnan(value) else f"{value:.4g}"

    def reason(self, model: Any, feature: str, value: float, record: Mapping[str, Any] | None) -> str | None:
        pct = None if np.isnan(value) else model.feature_builder.percentile(feature, value)
        return generic_reason(
            self.feature_description(feature), self.display_value(feature, value, record), pct
        )

    def entity_keys(self, record: Mapping[str, Any]) -> dict[str, str]:
        """Hashed keys used to link transactions that share an entity (card, device, ...)."""
        return {}

    def link_labels(self) -> dict[str, tuple[str, str]]:
        """entity type -> (label, strength)."""
        return {}

    # ------------------------------------------------------------------ data sources
    def replay_items(self, scenario: str, count: int, seed: int) -> tuple[list[ReplayItem], list[str]]:
        raise NotImplementedError

    def on_replay_end(self, processed: int) -> None:
        """Called when a replay finishes with the number of rows processed (stream cursors)."""

    def held_out(self) -> tuple[Any, np.ndarray, np.ndarray, float]:
        """(model input for the labelled held-out window, labels, amounts, window hours)."""
        raise NotImplementedError


def entity_hash(entity_type: str, key: tuple[Any, ...]) -> str:
    canonical = json.dumps([entity_type, [json_value(v) for v in key]])
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]
