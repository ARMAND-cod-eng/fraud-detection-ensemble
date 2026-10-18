"""IEEE-CIS ("ieee") serving profile: authorisation-time fields + online velocity features.

Online state is point-in-time correct: each new transaction reads its features *before* it is
added (``OnlineFeatureStore.score_and_update``), exactly as the offline training features were
computed (``scripts/check_feature_parity.py``: 0 mismatches on the real test window).

* Idempotent retries never reach the store: the service filters known transaction IDs first.
* Replays run with a single worker, so events reach the store in time order.
* Late events (event time before the stream watermark) are processed at the watermark and flagged.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.features.velocity import OnlineFeatureStore, compute_offline, entity_key
from fraudshield.profiles.base import (
    Record,
    ReplayItem,
    ServingProfile,
    entity_hash,
    is_missing,
    json_value,
)
from fraudshield.profiles.ieee import (
    AMOUNT,
    CATEGORICAL_FIELDS,
    CONTRACT_FIELDS,
    FEATURE_DESCRIPTIONS,
    IDENTIFIER_FIELDS,
    LABEL,
    REASON_TEMPLATES,
    TIME,
    VELOCITY_FEATURES,
    IEEEPreprocessor,
    reason_text,
)

ATTRIBUTION = (
    "Data: IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle competition). Used locally under the "
    "competition rules; the data and models trained on it are not redistributed or hosted."
)
SCENARIOS: dict[str, str] = {
    "historical": "Held-out test window streamed in time order through the online feature store, "
    "continuing where the previous replay stopped (natural fraud rate ~3.4%)",
}
_CARD = ("card1", "card2", "card3", "card5")
LINK_TYPES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "card": (_CARD, "Same card", "strong"),
    "card_email": ((*_CARD, "P_emaildomain"), "Same card and purchaser e-mail domain", "strong"),
    "card_addr": ((*_CARD, "addr1"), "Same card and billing region", "strong"),
    "device": (("DeviceInfo",), "Same device model (shared by many customers: weak link)", "weak"),
}


class IEEEProfile(ServingProfile):
    name = "ieee"
    preprocessor_kind = "ieee_v2"
    raw_fields = (TIME, *CONTRACT_FIELDS)
    time_field = TIME
    amount_field = AMOUNT
    label_field = LABEL
    currency = "USD"
    data_attribution = ATTRIBUTION
    scenarios = SCENARIOS
    ordered_stream = True
    linked_activity = True

    def __init__(self, replay_file: Path | None, warmup_file: Path | None) -> None:
        self.replay_file, self.warmup_file = replay_file, warmup_file
        self.velocity = True
        self.store = OnlineFeatureStore()
        self.watermark = -math.inf
        self.late_events = 0
        self.warm_up_events = 0
        self._lock = threading.Lock()
        self._cursor = 0
        self._pending_start = 0
        self._test: pd.DataFrame | None = None
        self._categories: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------ lifecycle
    def check_model(self, model: Any) -> None:
        super().check_model(model)
        prep: IEEEPreprocessor = model.feature_builder
        self.velocity = bool(prep.velocity)
        self._categories = {f: frozenset(v) for f, v in prep.categories.items()}

    def _reset_store(self) -> None:
        store, n, watermark = OnlineFeatureStore(), 0, -math.inf
        if self.velocity and self.warmup_file is not None and Path(self.warmup_file).exists():
            warm = pd.read_csv(self.warmup_file)
            n = store.warm_up(warm[[TIME, *CONTRACT_FIELDS]].to_dict("records"))
            watermark = float(warm[TIME].max()) if n else -math.inf
        self.store, self.warm_up_events, self.watermark = store, n, watermark

    def startup(self) -> dict[str, Any]:
        with self._lock:
            self._reset_store()
        return {"warm_up_events": self.warm_up_events, "entities": self.store.entity_count()}

    def status(self) -> dict[str, Any]:
        return {
            "profile": self.name,
            "online_feature_store": {
                "enabled": self.velocity,
                "backend": "in-process (single instance)",
                "features": len(VELOCITY_FEATURES) if self.velocity else 0,
                "entities": self.store.entity_count(),
                "events_seen": self.store.events_seen,
                "warm_up_events": self.warm_up_events,
                "watermark": None if math.isinf(self.watermark) else self.watermark,
                "late_events": self.late_events,
            },
            "replay_cursor": self._cursor,
        }

    # ------------------------------------------------------------------ request path
    def prepare(self, records: list[Record]) -> tuple[list[Record], list[list[str]]]:
        if not self.velocity:
            return records, [[] for _ in records]
        out: list[Record] = []
        warnings: list[list[str]] = []
        with self._lock:
            for rec in records:
                t, w = float(rec[TIME]), []
                event = rec
                if t < self.watermark:
                    self.late_events += 1
                    event = {**rec, TIME: self.watermark}
                    w.append(
                        f"event time {t:.0f} is before the stream watermark {self.watermark:.0f}; "
                        "velocity features computed as of the watermark"
                    )
                else:
                    self.watermark = t
                out.append({**rec, **self.store.score_and_update(event)})
                warnings.append(w)
        return out, warnings

    def base_features(self, model: Any, records: list[Record]) -> np.ndarray:
        prep: IEEEPreprocessor = model.feature_builder
        return np.vstack([prep.transform_record(r)[0] for r in records])

    def stored_features(self, record: Mapping[str, Any]) -> dict[str, Any]:
        out = super().stored_features(record)
        if self.velocity:
            out.update({f: json_value(record.get(f)) for f in VELOCITY_FEATURES})
        return out

    def warm_up_record(self) -> Record:
        return {TIME: 0.0, AMOUNT: 1.0}

    def quality_warnings(self, model: Any, base_row: np.ndarray, record: Mapping[str, Any]) -> list[str]:
        warnings: list[str] = []
        for f in CATEGORICAL_FIELDS:
            v = record.get(f)
            if isinstance(v, str) and v not in self._categories.get(f, frozenset()):
                warnings.append(f"{f}={v!r} not seen in training (treated as missing)")
        prof = model.feature_builder.profile
        for j, name in enumerate(model.feature_builder.feature_names):
            p, v = prof.get(name), base_row[j]
            if name in CATEGORICAL_FIELDS or not p or np.isnan(v):
                continue
            if not (p["min"] <= v <= p["max"]):
                warnings.append(f"{name}={v:.4g} outside training range [{p['min']:.4g}, {p['max']:.4g}]")
        return warnings[:10]

    # ------------------------------------------------------------------ analyst context
    def feature_description(self, feature: str) -> str:
        return FEATURE_DESCRIPTIONS.get(feature, feature)

    def display_value(self, feature: str, value: float, record: Mapping[str, Any] | None) -> str:
        if feature in CATEGORICAL_FIELDS:
            raw = None if record is None else record.get(feature)
            return "missing" if is_missing(raw) else str(raw)
        if np.isnan(value):
            return "missing"
        if feature == AMOUNT:
            return f"{value:,.2f} USD"
        if feature == "email_domains_match":
            return "yes" if value else "no"
        if feature in IDENTIFIER_FIELDS or feature == "day_of_week":
            return f"{value:.0f}"
        if feature == "hour_of_day":
            return f"{value:.1f}"
        return f"{value:,.4g}"

    def reason(self, model: Any, feature: str, value: float, record: Mapping[str, Any] | None) -> str | None:
        if feature in REASON_TEMPLATES:
            text = reason_text(feature, value, model.feature_builder.typical(feature))
            if text is not None:
                return text
        return super().reason(model, feature, value, record)

    def entity_keys(self, record: Mapping[str, Any]) -> dict[str, str]:
        out = {}
        for kind, (fields, _, _) in LINK_TYPES.items():
            key = entity_key(record, fields)
            if key is not None:
                out[kind] = entity_hash(kind, key)
        return out

    def link_labels(self) -> dict[str, tuple[str, str]]:
        return {k: (label, strength) for k, (_, label, strength) in LINK_TYPES.items()}

    # ------------------------------------------------------------------ data sources
    def _test_frame(self) -> pd.DataFrame:
        if self._test is None:
            if self.replay_file is None or not Path(self.replay_file).exists():
                raise FileNotFoundError("IEEE replay file not found; run `fraudshield train-ieee` first")
            self._test = pd.read_csv(self.replay_file)
        return self._test

    def replay_items(self, scenario: str, count: int, seed: int) -> tuple[list[ReplayItem], list[str]]:
        if scenario not in self.scenarios:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(self.scenarios)}")
        df = self._test_frame()
        notes = [f"source: held-out test window ({Path(str(self.replay_file)).name})"]
        with self._lock:
            if self._cursor >= len(df):
                self._cursor = 0
                self._reset_store()
                notes.append("end of the held-out window reached: feature store re-warmed, stream restarted")
            start = self._pending_start = self._cursor
        sample = df.iloc[start : start + count]
        notes.append(
            f"rows {start:,}-{start + len(sample) - 1:,} of {len(df):,} in time order; velocity features "
            "computed online (the stream order is fixed, so the seed is unused)"
        )
        items: list[ReplayItem] = [
            (rec, int(label), False)
            for rec, label in zip(
                sample[[TIME, *CONTRACT_FIELDS]].to_dict("records"), sample[LABEL].tolist(), strict=True
            )
        ]
        return items, notes

    def on_replay_end(self, processed: int) -> None:
        with self._lock:
            self._cursor = self._pending_start + processed

    def held_out(self) -> tuple[Any, np.ndarray, np.ndarray, float]:
        test = self._test_frame()
        frame = test
        if self.velocity:
            if self.warmup_file is None or not Path(self.warmup_file).exists():
                raise FileNotFoundError("IEEE warm-up file not found; run `fraudshield train-ieee` first")
            history = pd.concat([pd.read_csv(self.warmup_file), test], ignore_index=True)
            feats = compute_offline(history).iloc[-len(test) :].reset_index(drop=True)
            frame = pd.concat([test.reset_index(drop=True), feats], axis=1)
        hours = float(test[TIME].max() - test[TIME].min()) / 3600
        return frame, test[LABEL].to_numpy(), test[AMOUNT].to_numpy(), hours
