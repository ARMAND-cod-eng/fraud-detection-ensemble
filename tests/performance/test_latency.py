"""Latency guards. Opt-in (timing depends on hardware): FRAUDSHIELD_RUN_PERF=1 pytest tests/performance"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from fraudshield.data.schema import RAW_FEATURE_COLUMNS
from fraudshield.data.synthetic import generate_creditcard_like
from tests.conftest import txn_payload

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(os.environ.get("FRAUDSHIELD_RUN_PERF") != "1", reason="set FRAUDSHIELD_RUN_PERF=1"),
]


def test_in_process_single_row_p95_under_budget(trained) -> None:
    from fraudshield.inference.scorer import FraudScorer

    scorer = FraudScorer.from_registry(trained["root"])
    rows = generate_creditcard_like(n=300, seed=11)
    arr = rows[list(RAW_FEATURE_COLUMNS)].to_numpy()
    times = []
    for i in range(300):
        cols = {c: arr[i, j : j + 1] for j, c in enumerate(RAW_FEATURE_COLUMNS)}
        t = time.perf_counter()
        scorer.score_columns(cols)
        times.append((time.perf_counter() - t) * 1000)
    assert np.percentile(times[20:], 95) < 10.0


def test_http_score_p95_under_budget(client) -> None:
    times = []
    for i in range(200):
        body = txn_payload(seed=i % 50)
        t = time.perf_counter()
        r = client.post("/api/v1/transactions/score", json=body)
        times.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200
    assert np.percentile(times[20:], 95) < 50.0
