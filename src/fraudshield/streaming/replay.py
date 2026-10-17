"""Controlled transaction replay into the scoring service.

Scenarios
* ``historical``  - the held-out test window (real, labelled, never used for training) in
  chronological order. Labels are stored as delayed ground truth, never shown to the model.
* ``enriched``    - real held-out transactions resampled so ~5% are (real) frauds. Makes the demo
  eventful; alert rates are inflated by construction and the UI says so.
* ``synthetic_drift`` - held-out rows with an injected shift in Amount and several components.
  Rows are marked ``synthetic=True`` and carry no ground truth.

Reproducibility: the row order and every perturbation derive from ``seed``.
Backpressure: a bounded queue sits between the pacing producer and scoring workers; when
scoring falls behind, the producer waits instead of buffering without limit. Repeated scoring
failures stop the run in a ``failed`` state instead of retrying forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraudshield.data.schema import LABEL_COLUMN, RAW_FEATURE_COLUMNS, V_COLUMNS
from fraudshield.data.synthetic import generate_creditcard_like
from fraudshield.service import FraudService, ScoreRequest

Scenario = Literal["historical", "enriched", "synthetic_drift"]
SCENARIOS: dict[str, str] = {
    "historical": "Held-out test window, chronological, natural fraud rate (~0.13%)",
    "enriched": "Held-out transactions resampled to up to 5% real frauds, each at most once (inflated alert rate)",
    "synthetic_drift": "SYNTHETIC: held-out rows with injected Amount/component shift (drift demo)",
}


@dataclass
class ReplayStatus:
    state: str = "idle"  # idle | running | paused | completed | stopped | failed
    run_id: str | None = None
    scenario: str | None = None
    seed: int | None = None
    rate_tps: float = 0.0
    target_count: int = 0
    produced: int = 0
    scored: int = 0
    failed: int = 0
    alerts: int = 0
    backpressure_waits: int = 0
    queue_depth: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    last_error: str | None = None
    data_source: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
        d["elapsed_s"] = elapsed
        d["effective_tps"] = self.scored / elapsed if elapsed > 0 else 0.0
        return d


class ReplayEngine:
    def __init__(
        self,
        service: FraudService,
        replay_file: Path | None,
        queue_size: int = 200,
        workers: int = 2,
        max_consecutive_failures: int = 20,
        timeout_s: float = 5.0,
    ) -> None:
        self.service = service
        self.replay_file = replay_file
        self.queue_size = queue_size
        self.workers = workers
        self.max_consecutive_failures = max_consecutive_failures
        self.timeout_s = timeout_s
        self.status = ReplayStatus()
        self._tasks: list[asyncio.Task[None]] = []
        self._resume = asyncio.Event()
        self._stop = asyncio.Event()
        self._frame: pd.DataFrame | None = None

    # ------------------------------------------------------------------ data
    def _source(self) -> tuple[pd.DataFrame, str]:
        if self._frame is None:
            if self.replay_file is not None and Path(self.replay_file).exists():
                self._frame = pd.read_csv(self.replay_file)
                self._source_name = f"held-out test window ({Path(self.replay_file).name})"
            else:
                self._frame = generate_creditcard_like(n=20_000, fraud_rate=0.01, seed=7)
                self._source_name = "SYNTHETIC fallback (replay file not found)"
        return self._frame, self._source_name

    def build_rows(self, scenario: str, count: int, seed: int) -> tuple[list[ScoreRequest], list[str]]:
        df, source_name = self._source()
        rng = np.random.default_rng(seed)
        notes = [f"source: {source_name}"]
        synthetic_source = source_name.startswith("SYNTHETIC")
        if scenario == "historical":
            start = int(rng.integers(0, max(1, len(df) - count))) if len(df) > count else 0
            sample = df.iloc[start : start + count]
            notes.append(f"contiguous chronological slice starting at row {start}")
        elif scenario == "enriched":
            fraud = df[df[LABEL_COLUMN] == 1]
            legit = df[df[LABEL_COLUMN] == 0]
            # Each real fraud appears at most once: long runs get a lower fraud share instead of copies.
            n_fraud = max(1, min(round(count * 0.05), len(fraud)))
            idx_f = rng.choice(len(fraud), size=n_fraud, replace=False)
            idx_l = rng.choice(len(legit), size=min(count - n_fraud, len(legit)), replace=False)
            sample = pd.concat([fraud.iloc[idx_f], legit.iloc[idx_l]]).sort_values("Time", kind="stable")
            notes.append(
                f"{n_fraud} distinct real frauds among {len(sample)} rows ({n_fraud / len(sample):.1%})"
            )
        elif scenario == "synthetic_drift":
            sample = df.iloc[rng.choice(len(df), size=min(count, len(df)), replace=count > len(df))].copy()
            sample["Amount"] = (sample["Amount"] * rng.uniform(2.0, 3.5, len(sample))).round(2)
            for c in V_COLUMNS[:6]:
                sample[c] = sample[c] + 1.5 + rng.normal(0, 0.3, len(sample))
            notes.append("Amount x2-3.5 and V1-V6 shifted by +1.5: SYNTHETIC rows, no ground truth")
        else:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        has_labels = scenario != "synthetic_drift" and not synthetic_source
        reqs = []
        for rec, label in zip(
            sample[list(RAW_FEATURE_COLUMNS)].to_dict("records"), sample[LABEL_COLUMN].tolist(), strict=True
        ):
            reqs.append(
                ScoreRequest(
                    features=rec,
                    ground_truth=int(label) if has_labels else None,
                    synthetic=scenario == "synthetic_drift" or synthetic_source,
                )
            )
        return reqs, notes

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return self.status.state in {"running", "paused"}

    async def start(self, scenario: str, rate_tps: float, count: int, seed: int = 42) -> ReplayStatus:
        if self.running:
            raise RuntimeError("a replay is already running; stop it first")
        rows, notes = self.build_rows(scenario, count, seed)
        run_id = uuid.uuid4().hex[:8]
        self.status = ReplayStatus(
            state="running",
            run_id=run_id,
            scenario=scenario,
            seed=seed,
            rate_tps=rate_tps,
            target_count=len(rows),
            started_at=time.time(),
            data_source=notes[0],
            notes=notes,
        )
        for i, r in enumerate(rows):
            r.transaction_id = f"rp-{run_id}-{i:06d}"
        self._resume.set()
        self._stop.clear()
        queue: asyncio.Queue[ScoreRequest | None] = asyncio.Queue(maxsize=self.queue_size)
        self._tasks = [asyncio.create_task(self._produce(rows, queue))]
        self._tasks += [asyncio.create_task(self._work(queue)) for _ in range(self.workers)]
        self.service.metrics.replay_running.set(1)
        return self.status

    def pause(self) -> ReplayStatus:
        if self.status.state == "running":
            self._resume.clear()
            self.status.state = "paused"
        return self.status

    def resume(self) -> ReplayStatus:
        if self.status.state == "paused":
            self._resume.set()
            self.status.state = "running"
        return self.status

    def set_rate(self, rate_tps: float) -> ReplayStatus:
        self.status.rate_tps = rate_tps
        return self.status

    async def stop(self, state: str = "stopped") -> ReplayStatus:
        if self.running:
            self._stop.set()
            self._resume.set()
            for t in self._tasks:
                t.cancel()
            for t in self._tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            self._finish(state)
        return self.status

    def _finish(self, state: str) -> None:
        self.status.state = state
        self.status.finished_at = time.time()
        self.status.queue_depth = 0
        self.service.metrics.replay_running.set(0)

    # ------------------------------------------------------------------ loops
    async def _produce(self, rows: list[ScoreRequest], queue: asyncio.Queue[ScoreRequest | None]) -> None:
        tick = 0.05
        credit = 0.0
        i = 0
        try:
            while i < len(rows) and not self._stop.is_set():
                await self._resume.wait()
                credit += self.status.rate_tps * tick
                while credit >= 1 and i < len(rows):
                    if queue.full():
                        self.status.backpressure_waits += 1
                    await queue.put(rows[i])  # blocks when workers fall behind (backpressure)
                    i += 1
                    credit -= 1
                    self.status.produced = i
                self.status.queue_depth = queue.qsize()
                await asyncio.sleep(tick)
        finally:
            for _ in range(self.workers):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(None)

    async def _work(self, queue: asyncio.Queue[ScoreRequest | None]) -> None:
        consecutive = 0
        loop = asyncio.get_running_loop()
        while True:
            item = await queue.get()
            if item is None:
                break
            batch = [item]
            while len(batch) < 50 and not queue.empty():
                nxt = queue.get_nowait()
                if nxt is None:
                    queue.put_nowait(None)
                    break
                batch.append(nxt)
            try:
                res = await asyncio.wait_for(
                    loop.run_in_executor(None, self.service.score, batch, "replay", self.status.scenario),
                    timeout=self.timeout_s,
                )
                self.status.scored += len(res)
                self.status.alerts += sum(r["recommended_action"] != "approve" for r in res)
                consecutive = 0
            except Exception as exc:  # isolate failures; never crash the API process
                self.status.failed += len(batch)
                self.status.last_error = type(exc).__name__
                self.service.monitor.record_error("replay_" + type(exc).__name__)
                consecutive += 1
                if consecutive >= self.max_consecutive_failures:
                    self._finish("failed")
                    self._stop.set()
                    return
            self.status.queue_depth = queue.qsize()
            if self.status.scored + self.status.failed >= self.status.target_count:
                self._finish("completed")
                return
