"""The "ieee" serving profile end to end: online velocity features, idempotency, linked activity,
reason codes, ordered replay and policy simulation.

A tiny v2 model is trained on *synthetic*, schema-shaped data (never the Kaggle files); its
metrics are meaningless and never reported.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ieee_trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from fraudshield.data.ieee_cis import synthetic_ieee_like
    from fraudshield.training.ieee_v2 import run_ieee_v2

    root = tmp_path_factory.mktemp("ieee")
    cfg = yaml.safe_load((ROOT / "configs" / "ieee_v2.yaml").read_text())
    cfg["gate"].update(min_test_pr_auc=0.0, min_test_roc_auc=0.0)  # synthetic data: exercise the path
    out = run_ieee_v2(
        cfg,
        frame=synthetic_ieee_like(n=4000, seed=0),
        artifacts_root=root,
        results_dir=None,
        quick=True,
        force_variant="raw_plus_velocity",  # the serving path under test needs velocity features
    )
    assert out["evaluation"]["promotion"]["promoted"]
    return {"root": root, **out}


@pytest.fixture
def ieee_client(ieee_trained: dict[str, Any], tmp_path: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from api.main import create_app
    from fraudshield.config import Settings

    root = Path(ieee_trained["root"])
    settings = Settings(
        profile="ieee",
        artifacts_dir=root,
        db_path=tmp_path / "ieee.db",
        replay_file=root / "replay" / "ieee_test_window.csv.gz",
        warmup_file=root / "replay" / "ieee_warmup.csv.gz",
        dashboard_dir=None,
        log_json=False,
        log_level="WARNING",
        policy_path=ROOT / "configs" / "decision_policy.yaml",
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        yield c


def _txn(tid: str, t: float, amount: float = 120.0, card1: int = 4242, **extra: Any) -> dict[str, Any]:
    body = {
        "transaction_id": tid,
        "time_seconds": t,
        "amount": amount,
        "ProductCD": "W",
        "card1": card1,
        "card2": 321.0,
        "card3": 150.0,
        "card5": 226.0,
        "card4": "visa",
        "card6": "debit",
        "addr1": 204.0,
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "dev7",
    }
    return {**body, **extra}


def _online(client: Any) -> dict[str, Any]:
    return client.get("/api/v1/models/info").json()["online"]["online_feature_store"]


def test_config_ready_and_profile_contract(ieee_client) -> None:
    cfg = ieee_client.get("/api/v1/config").json()
    assert cfg["profile"] == "ieee" and cfg["currency"] == "USD" and cfg["linked_activity"] is True
    assert list(cfg["scenarios"]) == ["historical"] and "IEEE-CIS" in cfg["data_attribution"]
    assert ieee_client.get("/ready").status_code == 200
    info = ieee_client.get("/api/v1/models/info").json()
    assert info["profile"] == "ieee" and _online(ieee_client)["warm_up_events"] > 0
    # The OpenAPI schema documents the IEEE contract, not the credit-card one.
    schema = ieee_client.get("/openapi.json").json()["components"]["schemas"]
    assert "IEEETransactionIn" in schema and "TransactionIn" not in schema
    # A credit-card payload is rejected by this deployment.
    bad = {"amount": 1.0, "time_seconds": 1.0, "features": {"V1": 0.0}}
    assert ieee_client.post("/api/v1/transactions/score", json=bad).status_code == 422


def test_velocity_is_point_in_time_and_idempotent(ieee_client) -> None:
    t0 = _online(ieee_client)["watermark"] + 10
    first = ieee_client.post("/api/v1/transactions/score", json=_txn("v-1", t0)).json()
    assert first["idempotent_replay"] is False
    seen = _online(ieee_client)["events_seen"]
    # A retry with the same ID and payload is answered from the store and never re-counted.
    again = ieee_client.post("/api/v1/transactions/score", json=_txn("v-1", t0)).json()
    assert again["idempotent_replay"] is True and _online(ieee_client)["events_seen"] == seen
    conflict = ieee_client.post("/api/v1/transactions/score", json=_txn("v-1", t0, amount=999.0))
    assert conflict.status_code == 409

    ieee_client.post("/api/v1/transactions/score", json=_txn("v-2", t0 + 60, amount=240.0))
    f1 = ieee_client.get("/api/v1/transactions/v-1").json()["features"]
    f2 = ieee_client.get("/api/v1/transactions/v-2").json()["features"]
    assert f2["v_card_count_1h"] == f1["v_card_count_1h"] + 1  # v-1 precedes v-2 within the hour
    assert f2["v_card_secs_since_prev"] == pytest.approx(60.0)
    assert f2["v_card_amount_24h"] == pytest.approx(f1["v_card_amount_24h"] + 120.0)
    # Missing optional fields are stored as null (JSON-safe), not NaN.
    assert f1["dist1"] is None and f1["R_emaildomain"] is None


def test_late_event_and_unseen_category_are_flagged(ieee_client) -> None:
    wm = _online(ieee_client)["watermark"]
    late = ieee_client.post(
        "/api/v1/transactions/score", json=_txn("late-1", wm - 5000, DeviceInfo="never-seen-device")
    ).json()
    warnings = " | ".join(late["data_quality_warnings"])
    assert "watermark" in warnings and "not seen in training" in warnings


def test_linked_activity_and_reason_codes(ieee_client) -> None:
    t0 = _online(ieee_client)["watermark"] + 10
    for i in range(4):
        ieee_client.post("/api/v1/transactions/score", json=_txn(f"l-{i}", t0 + i * 30, card1=777))
    linked = ieee_client.get("/api/v1/transactions/l-3/linked").json()
    assert linked["available"] is True
    card = next(x for x in linked["links"] if x["type"] == "card")
    assert card["strength"] == "strong" and card["earlier"] == 3 and card["total"] >= 3
    assert {t["transaction_id"] for t in card["transactions"]} >= {"l-0", "l-1", "l-2"}
    assert ieee_client.get("/api/v1/transactions/missing-1/linked").status_code == 404

    exp = ieee_client.get("/api/v1/transactions/l-3/explanation").json()["explanation"]
    assert exp["top_factors"] and all(f["reason"] for f in exp["top_factors"])
    assert all("display_value" in f for f in exp["top_factors"])
    cats = [f for f in exp["top_factors"] if f["feature"] in ("card4", "card6", "P_emaildomain", "ProductCD")]
    assert all(f["display_value"] in ("visa", "debit", "gmail.com", "W") for f in cats)  # raw text, not codes


def test_ordered_replay_continues_the_stream(ieee_client) -> None:
    assert (
        ieee_client.post("/api/v1/replay/start", json={"scenario": "enriched", "count": 10}).status_code
        == 422
    )

    def run(count: int) -> dict[str, Any]:
        r = ieee_client.post(
            "/api/v1/replay/start", json={"scenario": "historical", "rate_tps": 1000, "count": count}
        )
        assert r.status_code == 200
        deadline = time.time() + 30
        while time.time() < deadline:
            st = ieee_client.get("/api/v1/replay/status").json()
            if st["state"] == "completed":
                return st
            time.sleep(0.1)
        raise AssertionError(st)

    first = run(40)
    assert first["scored"] == 40 and first["failed"] == 0 and "rows 0-39" in first["notes"][1]
    second = run(25)
    assert "rows 40-64" in second["notes"][1]  # the stream continues where it stopped
    mon = ieee_client.get("/api/v1/monitoring").json()
    assert mon["labelled_performance"]["n"] >= 65 and mon["drift"]["available"] is True
    sim = ieee_client.post(
        "/api/v1/policy/simulate", json={"review_threshold": 0.2, "decline_threshold": 0.9}
    ).json()
    assert sim["result"] and sim["capacity"]


def test_profile_mismatch_is_reported(client_factory) -> None:
    """A credit-card artifact cannot be served under FRAUDSHIELD_PROFILE=ieee."""
    c = client_factory(profile="ieee")
    ready = c.get("/ready")
    assert ready.status_code == 503 and ready.json()["reason"] == "RuntimeError"
    assert c.get("/api/v1/config").json()["profile"] == "ieee"


def test_creditcard_profile_has_no_linked_activity(client) -> None:
    from tests.conftest import txn_payload

    client.post("/api/v1/transactions/score", json=txn_payload(seed=3, transaction_id="cc-1"))
    linked = client.get("/api/v1/transactions/cc-1/linked").json()
    assert linked["available"] is False and linked["links"] == []
    assert client.get("/api/v1/config").json()["profile"] == "creditcard"
