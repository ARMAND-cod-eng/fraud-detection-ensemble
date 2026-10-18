"""Velocity (entity-history) features with one definition and two engines.

A real-time fraud model needs features such as "transactions on this card in the last 24 hours".
They must be computed *identically* in training (batch, over history) and in serving (one event at
a time, before the event is stored). Otherwise the model is trained on numbers it never sees in
production: train/serve skew, the defect of the original notebook.

* ``VELOCITY_SPECS`` declares every feature once.
* ``compute_offline`` evaluates the specs over a time-ordered frame (vectorised).
* ``OnlineFeatureStore`` evaluates the same specs per event from bounded in-memory state.
* ``tests/unit/test_velocity.py`` and ``scripts/check_feature_parity.py`` prove the two agree.

Semantics, identical in both engines, for event *i* at time ``t_i`` with entity key ``k_i``:
the window is [t_i - W, t_i] over **earlier** events in stream order (ties at the same second count
only if they arrived earlier). Every aggregate is windowed, including "seconds since previous", so
state is bounded and values stay comparable across time (cumulative counts drift, see
``docs/ML_EVALUATION.md`` §11). If any entity field is missing the feature is NaN, and missing
values of a ``distinct`` field are not counted.

Assumption (documented limitation): events arrive in non-decreasing time order per entity; a late
event is evaluated as of its arrival.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

HOUR, DAY = 3600, 86_400
AGGREGATIONS = ("count", "sum_amount", "distinct", "secs_since_prev", "amount_ratio_to_mean")
# Window totals at or below this are treated as zero by both engines: running sums accumulate
# floating-point residue that would otherwise turn "no spend" into a tiny positive mean.
MIN_TOTAL = 1e-6
CARD = ("card1", "card2", "card3", "card5")


@dataclass(frozen=True)
class VelocitySpec:
    name: str
    entity: tuple[str, ...]
    window_s: int
    agg: str
    field: str | None = None  # for "distinct"
    template: str = ""  # plain-language reason-code sentence; {value} is the feature value

    def __post_init__(self) -> None:
        if self.agg not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {self.agg!r}")
        if (self.agg == "distinct") != (self.field is not None):
            raise ValueError("'distinct' needs a field, and only 'distinct' takes one")


# Pre-registered before the v2 experiments (docs/ML_EVALUATION.md §12). Windowed only.
VELOCITY_SPECS: tuple[VelocitySpec, ...] = (
    VelocitySpec(
        "v_card_count_1h",
        CARD,
        HOUR,
        "count",
        template="{value:.0f} earlier transactions on this card in the last hour",
    ),
    VelocitySpec(
        "v_card_count_24h",
        CARD,
        DAY,
        "count",
        template="{value:.0f} earlier transactions on this card in the last 24 h",
    ),
    VelocitySpec(
        "v_card_count_7d",
        CARD,
        7 * DAY,
        "count",
        template="{value:.0f} earlier transactions on this card in the last 7 days",
    ),
    VelocitySpec(
        "v_card_amount_24h",
        CARD,
        DAY,
        "sum_amount",
        template="{value:,.2f} spent on this card in the last 24 h",
    ),
    VelocitySpec(
        "v_card_amount_ratio_7d",
        CARD,
        7 * DAY,
        "amount_ratio_to_mean",
        template="amount is {value:.1f}x this card's 7-day average",
    ),
    VelocitySpec(
        "v_card_secs_since_prev",
        CARD,
        7 * DAY,
        "secs_since_prev",
        template="{value:,.0f} seconds since this card's previous transaction",
    ),
    VelocitySpec(
        "v_card_distinct_email_7d",
        CARD,
        7 * DAY,
        "distinct",
        "P_emaildomain",
        template="{value:.0f} different e-mail domains used with this card in 7 days",
    ),
    VelocitySpec(
        "v_card_distinct_device_7d",
        CARD,
        7 * DAY,
        "distinct",
        "DeviceInfo",
        template="{value:.0f} different device models used with this card in 7 days",
    ),
    VelocitySpec(
        "v_card_distinct_addr_7d",
        CARD,
        7 * DAY,
        "distinct",
        "addr1",
        template="{value:.0f} different billing regions used with this card in 7 days",
    ),
    VelocitySpec(
        "v_card_email_count_7d",
        ("card1", "P_emaildomain"),
        7 * DAY,
        "count",
        template="{value:.0f} earlier transactions with this card and e-mail domain in 7 days",
    ),
    VelocitySpec(
        "v_device_count_1h",
        ("DeviceInfo",),
        HOUR,
        "count",
        template="{value:.0f} transactions from this device model in the last hour",
    ),
    VelocitySpec(
        "v_device_distinct_cards_24h",
        ("DeviceInfo",),
        DAY,
        "distinct",
        "card1",
        template="{value:.0f} different cards used with this device model in 24 h",
    ),
    VelocitySpec(
        "v_addr_distinct_cards_24h",
        ("addr1",),
        DAY,
        "distinct",
        "card1",
        template="{value:.0f} different cards billed to this region in 24 h",
    ),
    VelocitySpec(
        "v_email_count_1h",
        ("P_emaildomain",),
        HOUR,
        "count",
        template="{value:.0f} transactions from this e-mail domain in the last hour",
    ),
    VelocitySpec(
        "v_email_distinct_cards_24h",
        ("P_emaildomain",),
        DAY,
        "distinct",
        "card1",
        template="{value:.0f} different cards used with this e-mail domain in 24 h",
    ),
)


def _missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...] | None:
    vals = tuple(record.get(f) for f in fields)
    return None if any(_missing(v) for v in vals) else vals


entity_key = _key  # public: the same keying is used for linked activity in the serving profile


# --------------------------------------------------------------------------- offline engine
def window_counts(codes: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """For each row: number of earlier rows with the same code and time in [t - window, t].

    Vectorised: sort by (code, time, row) and encode each row as ``code * big + time``; one
    ``searchsorted`` then finds every window start without crossing into the previous code's block.
    """
    left, order = _window_left(codes, t, window)
    n = len(order)
    counts = np.empty(n, dtype=np.int64)
    counts[order] = np.arange(n) - left
    return counts


def _window_left(codes: np.ndarray, t: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    t64 = np.asarray(t, dtype=np.int64)
    order = np.lexsort((np.arange(n), t64, codes))
    c, tt = np.asarray(codes, dtype=np.int64)[order], t64[order]
    base = int(tt.min())
    big = int(tt.max()) - base + window + 1
    comp = c * big + (tt - base)
    return np.searchsorted(comp, comp - window, side="left"), order


def window_sums(codes: np.ndarray, t: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """For each row: sum of ``values`` over earlier rows with the same code within the window."""
    left, order = _window_left(codes, t, window)
    n = len(order)
    v = np.asarray(values, dtype=np.float64)[order]
    # Cumulative sums restart for every entity, so one entity's large history cannot swamp
    # another's small amounts through floating-point cancellation.
    excl = pd.Series(v).groupby(np.asarray(codes)[order]).cumsum().to_numpy() - v  # sum of earlier rows
    pos = np.arange(n)
    sums = np.empty(n, dtype=np.float64)
    sums[order] = excl[pos] - excl[left]
    return sums


def window_prev_gap(codes: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """Seconds since the previous row with the same code, NaN if none within the window."""
    left, order = _window_left(codes, t, window)
    n = len(order)
    tt = np.asarray(t, dtype=np.float64)[order]
    pos = np.arange(n)
    has_prev = pos - left > 0
    gap = np.full(n, np.nan)
    gap[has_prev] = tt[has_prev] - tt[pos[has_prev] - 1]
    out = np.empty(n, dtype=np.float64)
    out[order] = gap
    return out


def window_distinct(
    entity_codes: np.ndarray, other_codes: np.ndarray, t: np.ndarray, window: int
) -> np.ndarray:
    """For each row (time order): distinct ``other`` values (>= 0) earlier with the same entity in the window."""
    state: dict[int, tuple[deque[tuple[int, int]], dict[int, int]]] = {}
    out = np.empty(len(t), dtype=np.int64)
    rows = zip(entity_codes.tolist(), other_codes.tolist(), np.asarray(t).tolist(), strict=True)
    for i, (e, o, ti) in enumerate(rows):
        st = state.get(e)
        if st is None:
            st = state[e] = (deque(), {})
        dq, cnt = st
        while dq and dq[0][0] < ti - window:
            _, old = dq.popleft()
            if old >= 0:
                k = cnt[old] - 1
                if k:
                    cnt[old] = k
                else:
                    del cnt[old]
        out[i] = len(cnt)
        dq.append((ti, o))
        if o >= 0:
            cnt[o] = cnt.get(o, 0) + 1
    return out


def _codes(frame: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Integer code per entity key and a missing mask (any entity field missing)."""
    sub = frame[list(fields)]
    missing = sub.isna().any(axis=1).to_numpy()
    if len(fields) == 1:
        codes = pd.factorize(sub.iloc[:, 0])[0]
    else:
        codes = pd.MultiIndex.from_frame(sub.astype(object)).factorize()[0]
    return np.asarray(codes, dtype=np.int64), missing


def compute_offline(
    frame: pd.DataFrame,
    specs: Iterable[VelocitySpec] = VELOCITY_SPECS,
    time_field: str = "TransactionDT",
    amount_field: str = "TransactionAmt",
) -> pd.DataFrame:
    """Evaluate ``specs`` for every row of a frame in stream order (sorted by time, stable)."""
    if not frame[time_field].is_monotonic_increasing:
        raise ValueError("frame must be sorted by time (stable) to define stream order")
    t = frame[time_field].to_numpy()
    amount = frame[amount_field].to_numpy(dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    for s in specs:
        if s.entity not in cache:
            cache[s.entity] = _codes(frame, s.entity)
        codes, missing = cache[s.entity]
        if s.agg == "count":
            vals = window_counts(codes, t, s.window_s).astype(np.float64)
        elif s.agg == "sum_amount":
            vals = window_sums(codes, t, amount, s.window_s)
        elif s.agg == "secs_since_prev":
            vals = window_prev_gap(codes, t, s.window_s)
        elif s.agg == "amount_ratio_to_mean":
            n = window_counts(codes, t, s.window_s).astype(np.float64)
            total = window_sums(codes, t, amount, s.window_s)
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.where((n > 0) & (total > MIN_TOTAL), amount / (total / np.maximum(n, 1)), np.nan)
        else:  # distinct
            assert s.field is not None
            col = frame[s.field]
            other = pd.factorize(col)[0]  # NaN -> -1 (not counted)
            vals = window_distinct(codes, np.asarray(other, dtype=np.int64), t, s.window_s).astype(np.float64)
        out[s.name] = np.where(missing, np.nan, vals)
    return pd.DataFrame(out, index=frame.index)


# --------------------------------------------------------------------------- online engine
@dataclass
class _Window:
    """Trailing-window state for one entity key: events, running sums and distinct counters."""

    events: deque[tuple[float, float, dict[str, Any]]] = field(default_factory=deque)
    total: float = 0.0
    distinct: dict[str, dict[Any, int]] = field(default_factory=dict)

    def evict(self, t: float, window: int) -> None:
        while self.events and self.events[0][0] < t - window:
            _, amount, vals = self.events.popleft()
            self.total -= amount
            for f, v in vals.items():
                cnt = self.distinct[f]
                k = cnt[v] - 1
                if k:
                    cnt[v] = k
                else:
                    del cnt[v]
        if not self.events:
            self.total = 0.0  # avoid drift from repeated float subtraction

    def add(self, t: float, amount: float, vals: dict[str, Any]) -> None:
        self.events.append((t, amount, vals))
        self.total += amount
        for f, v in vals.items():
            cnt = self.distinct.setdefault(f, {})
            cnt[v] = cnt.get(v, 0) + 1


class OnlineFeatureStore:
    """Per-event evaluation of the same specs from bounded state (one backend: process memory).

    ``features(record)`` reads state as of *before* the event; ``update(record)`` then records it.
    ``score_and_update`` does both atomically. Production would put the same structures in Redis
    (sorted sets per key) so several API instances share state; see docs/ARCHITECTURE.md.
    """

    def __init__(
        self,
        specs: Iterable[VelocitySpec] = VELOCITY_SPECS,
        time_field: str = "TransactionDT",
        amount_field: str = "TransactionAmt",
    ) -> None:
        self.specs = tuple(specs)
        self.time_field, self.amount_field = time_field, amount_field
        self._groups: dict[tuple[tuple[str, ...], int], list[VelocitySpec]] = {}
        for s in self.specs:
            self._groups.setdefault((s.entity, s.window_s), []).append(s)
        self._state: dict[tuple[tuple[str, ...], int], dict[tuple[Any, ...], _Window]] = {
            g: {} for g in self._groups
        }
        self._lock = threading.Lock()
        self.events_seen = 0

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self.specs]

    def _distinct_fields(self, specs: list[VelocitySpec]) -> list[str]:
        return sorted({s.field for s in specs if s.field is not None})

    def _read(self, record: Mapping[str, Any]) -> dict[str, float]:
        t = float(record[self.time_field])
        amount = float(record[self.amount_field])
        out: dict[str, float] = {}
        for (entity, window), specs in self._groups.items():
            key = _key(record, entity)
            if key is None:
                out.update({s.name: math.nan for s in specs})
                continue
            w = self._state[(entity, window)].get(key)
            if w is not None:
                w.evict(t, window)
            n = len(w.events) if w else 0
            for s in specs:
                if s.agg == "count":
                    out[s.name] = float(n)
                elif s.agg == "sum_amount":
                    out[s.name] = w.total if w and n else 0.0
                elif s.agg == "secs_since_prev":
                    out[s.name] = t - w.events[-1][0] if w and n else math.nan
                elif s.agg == "amount_ratio_to_mean":
                    total = w.total if w and n else 0.0
                    out[s.name] = amount / (total / n) if n and total > MIN_TOTAL else math.nan
                else:
                    assert s.field is not None
                    out[s.name] = float(len(w.distinct.get(s.field, {}))) if w else 0.0
        return out

    def _write(self, record: Mapping[str, Any]) -> None:
        t = float(record[self.time_field])
        amount = float(record[self.amount_field])
        for (entity, window), specs in self._groups.items():
            key = _key(record, entity)
            if key is None:
                continue
            states = self._state[(entity, window)]
            w = states.get(key)
            if w is None:
                w = states[key] = _Window()
            w.evict(t, window)
            vals = {f: record.get(f) for f in self._distinct_fields(specs) if not _missing(record.get(f))}
            w.add(t, amount, vals)
        self.events_seen += 1

    def features(self, record: Mapping[str, Any]) -> dict[str, float]:
        with self._lock:
            return self._read(record)

    def update(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._write(record)

    def score_and_update(self, record: Mapping[str, Any]) -> dict[str, float]:
        with self._lock:
            feats = self._read(record)
            self._write(record)
            return feats

    def warm_up(self, records: Iterable[Mapping[str, Any]]) -> int:
        """Replay history (in time order) to rebuild state, e.g. the last max-window of events."""
        n = 0
        with self._lock:
            for r in records:
                self._write(r)
                n += 1
        return n

    def entity_count(self) -> int:
        return sum(len(v) for v in self._state.values())
