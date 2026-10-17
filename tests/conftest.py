"""Shared fixtures.

Tests never need the Kaggle dataset: a tiny model is trained on *synthetic* data with the same
schema (``fraudshield.data.synthetic``), which exercises the full train -> artifact -> API path.
Metrics from these fixtures are meaningless and are never reported.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("FRAUDSHIELD_DISABLE_MLFLOW", "1")

from fraudshield.data.schema import V_COLUMNS
from fraudshield.data.synthetic import generate_creditcard_like


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    generate_creditcard_like(n=6000, fraud_rate=0.02, seed=0).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory, synthetic_csv: Path) -> dict[str, Any]:
    from fraudshield.training.pipeline import load_config, run_training

    root = tmp_path_factory.mktemp("artifacts")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "training.yaml")
    out = run_training(
        cfg,
        artifacts_root=root,
        results_dir=None,
        data_path=synthetic_csv,
        promote=True,
        export_replay=True,
        make_figures=False,
        quick=True,
    )
    out["root"] = root
    return out


def make_settings(trained: dict[str, Any], tmp_path: Path, **overrides: Any) -> Any:
    from fraudshield.config import Settings

    root = Path(trained["root"])
    base = {
        "artifacts_dir": root,
        "db_path": tmp_path / "test.db",
        "replay_file": root / "replay" / "creditcard_test_window.csv.gz",
        "dashboard_dir": None,
        "log_json": False,
        "log_level": "WARNING",
        "policy_path": Path(__file__).resolve().parents[1] / "configs" / "decision_policy.yaml",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client_factory(trained: dict[str, Any], tmp_path: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from api.main import create_app

    clients: list[Any] = []

    def factory(**overrides: Any) -> Any:
        c = TestClient(
            create_app(make_settings(trained, tmp_path, **overrides)), raise_server_exceptions=False
        )
        c.__enter__()
        clients.append(c)
        return c

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(client_factory: Any) -> Any:
    return client_factory()


def txn_payload(seed: int = 0, fraud_like: bool = False, transaction_id: str | None = None) -> dict[str, Any]:
    row = generate_creditcard_like(n=200, fraud_rate=0.05, seed=seed)
    row = row[row["Class"] == (1 if fraud_like else 0)].iloc[0]
    body: dict[str, Any] = {
        "amount": float(row["Amount"]),
        "time_seconds": float(row["Time"]),
        "features": {v: float(row[v]) for v in V_COLUMNS},
    }
    if transaction_id:
        body["transaction_id"] = transaction_id
    return body


@pytest.fixture
def live_server(trained: dict[str, Any], tmp_path: Path) -> Iterator[str]:
    """A real uvicorn server in a background thread (needed for streaming responses)."""
    import socket
    import threading
    import time

    import uvicorn

    from api.main import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(make_settings(trained, tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
