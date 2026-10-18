"""Controlled transaction replay into the scoring service.

Scenarios come from the serving profile:
* creditcard: ``historical`` (held-out window, chronological), ``enriched`` (resampled to ~5% real
  frauds; inflated alert rate by construction) and ``synthetic_drift`` (injected shift, marked
  ``synthetic=True``, no ground truth).
* ieee: ``historical`` only - the held-out window streamed *in order* through the online feature
  store, continuing where the previous replay stopped. Resampling would break the entity histories
  that velocity features are computed from.

Labels are stored as delayed ground truth and never shown to the model.
Reproducibility: the row order and every perturbation derive from ``seed``.
Backpressure: a bounded queue sits between the pacing producer and scoring workers; when
scoring falls behind, the producer waits instead of buffering without limit. Repeated scoring
failures stop the run in a ``failed`` state instead of retrying forever. Profiles with stateful
online features use a single worker so events reach the feature store in time order.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fraudshield.profiles.creditcard import SCENARIOS
from fraudshield.service import FraudService, ScoreRequest

__all__ = ["SCENARIOS", "ReplayEngine", "ReplayStatus"]


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
        queue_size: int = 200,
        workers: int = 2,
        max_consecutive_failures: int = 20,
        timeout_s: float = 5.0,
    ) -> None:
        self.service = service
        self.profile = service.profile
        self.queue_size = queue_size
        self.workers = 1 if self.profile.ordered_stream else workers
        self.max_consecutive_failures = max_consecutive_failures
        self.timeout_s = timeout_s
        self.status = ReplayStatus()
        self._tasks: list[asyncio.Task[None]] = []
        self._resume = asyncio.Event()
        self._stop = asyncio.Event()

    @property
    def scenarios(self) -> dict[str, str]:
        return self.profile.scenarios

    # ------------------------------------------------------------------ data
    def build_rows(self, scenario: str, count: int, seed: int) -> tuple[list[ScoreRequest], list[str]]:
        items, notes = self.profile.replay_items(scenario, count, seed)
        reqs = [ScoreRequest(features=rec, ground_truth=label, synthetic=syn) for rec, label, syn in items]
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
        self.profile.on_replay_end(self.status.scored + self.status.failed)
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
