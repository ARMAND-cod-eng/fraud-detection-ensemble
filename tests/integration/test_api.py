"""End-to-end API tests against a model trained on synthetic data (see tests/conftest.py)."""

from __future__ import annotations

import time

import pytest

from tests.conftest import txn_payload

pytestmark = pytest.mark.integration


def test_health_ready_metrics(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["checks"] == {"model_loaded": True, "store": True, "policy": True}
    client.post("/api/v1/transactions/score", json=txn_payload())
    m = client.get("/metrics").text
    assert "fraudshield_predictions_total" in m and "fraudshield_inference_duration_seconds" in m


def test_score_contract(client) -> None:
    r = client.post("/api/v1/transactions/score", json=txn_payload(transaction_id="t-1"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transaction_id"] == "t-1"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["recommended_action"] in {"approve", "review", "decline"}
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"] and body["policy_version"] and body["scored_at"]
    assert body["explanation_available"] is True
    assert r.headers["X-Request-ID"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_fraud_like_transaction_is_flagged(client) -> None:
    body = client.post("/api/v1/transactions/score", json=txn_payload(seed=5, fraud_like=True)).json()
    assert body["recommended_action"] in {"review", "decline"}
    assert body["decision_reasons"]


def test_policy_drives_action(client) -> None:
    pol = client.get("/api/v1/policy").json()["config"]
    body = client.post("/api/v1/transactions/score", json=txn_payload()).json()
    p = body["fraud_probability"]
    expected = (
        "decline"
        if p >= pol["decline_threshold"]
        else "review"
        if p >= pol["review_threshold"]
        else "approve"
    )
    assert body["recommended_action"] == expected


@pytest.mark.parametrize(
    "mutate,loc",
    [
        (lambda b: b["features"].pop("V7"), "V7"),
        (lambda b: b.update(amount=-1), "amount"),
        (lambda b: b.update(unexpected=1), "unexpected"),
        (lambda b: b["features"].update(V1="abc"), "V1"),
        (lambda b: b.update(transaction_id="bad id with spaces"), "transaction_id"),
    ],
)
def test_validation_errors_use_envelope(client, mutate, loc) -> None:
    body = txn_payload()
    mutate(body)
    r = client.post("/api/v1/transactions/score", json=body)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert any(loc in [str(x) for x in d["loc"]] for d in err["details"])
    assert "Traceback" not in r.text


def test_non_finite_values_rejected(client) -> None:
    raw = (
        '{"amount": 10, "time_seconds": 1, "features": {'
        + ", ".join(f'"V{i}": {"NaN" if i == 3 else 0}' for i in range(1, 29))
        + "}}"
    )
    r = client.post("/api/v1/transactions/score", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_idempotency(client) -> None:
    body = txn_payload(transaction_id="idem-1")
    first = client.post("/api/v1/transactions/score", json=body).json()
    again = client.post("/api/v1/transactions/score", json=body).json()
    assert again["idempotent_replay"] is True and first["idempotent_replay"] is False
    assert again["fraud_probability"] == first["fraud_probability"]
    body["amount"] += 1
    conflict = client.post("/api/v1/transactions/score", json=body)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"


def test_batch(client) -> None:
    txns = [txn_payload(seed=i, transaction_id=f"b-{i}") for i in range(20)]
    r = client.post("/api/v1/transactions/batch", json={"transactions": txns})
    assert r.status_code == 200
    assert r.json()["count"] == 20
    dup = client.post("/api/v1/transactions/batch", json={"transactions": [txns[0], txns[0]]})
    assert dup.status_code == 422


def test_batch_size_limit(client_factory) -> None:
    c = client_factory(max_batch_size=5)
    r = c.post("/api/v1/transactions/batch", json={"transactions": [txn_payload(seed=i) for i in range(6)]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "batch_too_large"


def test_list_detail_explanation_and_review(client) -> None:
    client.post(
        "/api/v1/transactions/score", json=txn_payload(seed=5, fraud_like=True, transaction_id="case-1")
    )
    client.post("/api/v1/transactions/score", json=txn_payload(seed=6, transaction_id="ok-1"))
    alerts = client.get("/api/v1/transactions", params={"alerts_only": True}).json()
    assert any(i["transaction_id"] == "case-1" for i in alerts["items"])
    assert client.get("/api/v1/transactions", params={"search": "ok-"}).json()["total"] == 1

    detail = client.get("/api/v1/transactions/case-1").json()
    assert detail["review_status"] == "open"
    assert "ground_truth" not in detail and detail["ground_truth_revealed"] is None
    exp = client.get("/api/v1/transactions/case-1/explanation").json()
    factors = exp["explanation"]["top_factors"]
    assert factors and {"feature", "contribution", "training_percentile"} <= set(factors[0])
    assert "not evidence" in exp["explanation"]["disclaimer"]
    assert client.get("/api/v1/transactions/ok-1/explanation").json()["cached"] is False

    r = client.post(
        "/api/v1/transactions/case-1/review", json={"decision": "confirmed_fraud", "notes": "card reported"}
    )
    assert r.status_code == 200
    detail = client.get("/api/v1/transactions/case-1").json()
    assert detail["review_status"] == "confirmed_fraud"
    assert detail["history"][-1]["notes"] == "card reported"
    assert client.get("/api/v1/transactions/missing-id").status_code == 404


def test_policy_update_and_simulation(client) -> None:
    bad = client.put("/api/v1/policy", json={"review_threshold": 0.5, "decline_threshold": 0.4})
    assert bad.status_code == 422
    r = client.put("/api/v1/policy", json={"review_threshold": 0.2, "decline_threshold": 0.95})
    assert r.status_code == 200
    version = r.json()["version"]
    body = client.post("/api/v1/transactions/score", json=txn_payload(seed=9)).json()
    assert body["policy_version"] == version
    sim = client.post(
        "/api/v1/policy/simulate", json={"review_threshold": 0.2, "decline_threshold": 0.95}
    ).json()
    res = sim["result"]
    assert res["approved"] + res["reviewed"] + res["declined"] == res["transactions"]
    assert sim["capacity"]


def test_overview_monitoring_and_model_info(client) -> None:
    for i in range(5):
        client.post("/api/v1/transactions/score", json=txn_payload(seed=i))
    ov = client.get("/api/v1/overview").json()
    assert ov["stats"]["n"] == 5 and len(ov["activity"]) == 60
    mon = client.get("/api/v1/monitoring").json()
    assert mon["drift"]["available"] is True
    assert mon["drift"]["overall_status"] == "insufficient_data"  # 5 rows is not a drift sample
    info = client.get("/api/v1/models/info").json()
    assert info["model_version"] and info["feature_names"]
    ev = client.get("/api/v1/models/evaluation").json()
    assert "test" in ev and "curves_test" in ev
    assert client.get("/api/v1/models/versions").json()["production"] == info["model_version"]


def test_replay_lifecycle(client) -> None:
    r = client.post(
        "/api/v1/replay/start", json={"scenario": "historical", "rate_tps": 500, "count": 60, "seed": 1}
    )
    assert r.status_code == 200
    assert client.post("/api/v1/replay/start", json={"count": 5}).status_code == 409  # one run at a time
    deadline = time.time() + 20
    while time.time() < deadline:
        st = client.get("/api/v1/replay/status").json()
        if st["state"] == "completed":
            break
        time.sleep(0.1)
    assert st["state"] == "completed" and st["scored"] == 60 and st["failed"] == 0
    listed = client.get("/api/v1/transactions", params={"search": f"rp-{st['run_id']}"}).json()
    assert listed["total"] == 60
    mon = client.get("/api/v1/monitoring").json()
    assert mon["labelled_performance"]["n"] >= 60  # replayed history carries delayed ground truth

    client.post("/api/v1/replay/start", json={"scenario": "synthetic_drift", "rate_tps": 5, "count": 1000})
    assert client.post("/api/v1/replay/pause").json()["state"] == "paused"
    assert client.post("/api/v1/replay/resume").json()["state"] == "running"
    assert client.patch("/api/v1/replay", json={"rate_tps": 50}).json()["rate_tps"] == 50
    assert client.post("/api/v1/replay/stop").json()["state"] == "stopped"
    synthetic = client.get("/api/v1/transactions", params={"limit": 5}).json()["items"]
    assert all(i["synthetic"] for i in synthetic if i["scenario"] == "synthetic_drift")


def test_api_key_protects_mutations(client_factory) -> None:
    c = client_factory(api_key="s3cret")
    assert c.post("/api/v1/transactions/score", json=txn_payload()).status_code == 401
    ok = c.post("/api/v1/transactions/score", json=txn_payload(), headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    assert (
        c.put("/api/v1/policy", json={"review_threshold": 0.2, "decline_threshold": 0.9}).status_code == 401
    )
    assert c.get("/api/v1/overview").status_code == 200  # reads stay open in this demo configuration


def test_rate_limit(client_factory) -> None:
    c = client_factory(rate_limit_per_second=0.01, rate_limit_burst=3)
    codes = [c.post("/api/v1/transactions/score", json=txn_payload(seed=i)).status_code for i in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    r = c.post("/api/v1/transactions/score", json=txn_payload())
    assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1


def test_inference_timeout(client_factory) -> None:
    c = client_factory(inference_timeout_ms=100)
    service = c.app.state.fs.service
    original = service.score

    def slow(*a, **k):
        time.sleep(0.4)
        return original(*a, **k)

    service.score = slow
    r = c.post("/api/v1/transactions/score", json=txn_payload(transaction_id="slow-1"))
    assert r.status_code == 503 and r.json()["error"]["code"] == "inference_timeout"
    service.score = original
    time.sleep(0.6)  # the timed-out call still completes in the background ...
    again = c.post("/api/v1/transactions/score", json=txn_payload(transaction_id="slow-1")).json()
    assert again["idempotent_replay"] is True  # ... and a retry with the same ID is safe


def test_unhandled_error_is_sanitised(client) -> None:
    service = client.app.state.fs.service

    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    service.score = boom
    r = client.post("/api/v1/transactions/score", json=txn_payload())
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in r.text and r.json()["error"]["request_id"]


def test_service_without_model(client_factory, tmp_path) -> None:
    c = client_factory(artifacts_dir=tmp_path / "empty")
    assert c.get("/health").status_code == 200
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["model_loaded"] is False
    s = c.post("/api/v1/transactions/score", json=txn_payload())
    assert s.status_code == 503 and s.json()["error"]["code"] == "model_unavailable"


def test_sse_stream_delivers_events(live_server) -> None:
    import threading

    import httpx

    def produce() -> None:
        time.sleep(0.5)
        with httpx.Client(base_url=live_server) as c:
            for i in range(3):
                c.post("/api/v1/transactions/score", json=txn_payload(seed=40 + i))

    threading.Thread(target=produce, daemon=True).start()
    seen = []
    with httpx.Client(base_url=live_server, timeout=10) as c, c.stream("GET", "/api/v1/stream") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("event: transaction"):
                seen.append(line)
                if len(seen) == 3:
                    break
    assert len(seen) == 3


def test_public_demo_mode_limits(client_factory) -> None:
    c = client_factory(demo_mode=True, demo_max_replay_tps=100, demo_max_replay_count=5000)
    cfg = c.get("/api/v1/config").json()
    assert cfg["demo_mode"] is True and cfg["policy_changes_allowed"] is False
    assert "Open Database License" in cfg["data_attribution"]
    # state-changing admin actions are refused ...
    assert c.post("/api/v1/admin/reset").json()["error"]["code"] == "disabled_in_demo"
    r = c.put("/api/v1/policy", json={"review_threshold": 0.2, "decline_threshold": 0.9})
    assert r.status_code == 403
    # ... replays are capped ...
    too_fast = c.post("/api/v1/replay/start", json={"scenario": "historical", "rate_tps": 500, "count": 10})
    assert too_fast.status_code == 422 and too_fast.json()["error"]["code"] == "demo_limit"
    assert (
        c.post(
            "/api/v1/replay/start", json={"scenario": "historical", "rate_tps": 50, "count": 6000}
        ).status_code
        == 422
    )
    ok = c.post("/api/v1/replay/start", json={"scenario": "historical", "rate_tps": 50, "count": 20})
    assert ok.status_code == 200
    assert c.patch("/api/v1/replay", json={"rate_tps": 500}).status_code == 422
    c.post("/api/v1/replay/stop")
    # ... while scoring, what-if simulation and analyst reviews keep working.
    assert c.post("/api/v1/transactions/score", json=txn_payload(transaction_id="demo-1")).status_code == 200
    sim = c.post("/api/v1/policy/simulate", json={"review_threshold": 0.2, "decline_threshold": 0.9})
    assert sim.status_code == 200
    assert (
        c.post("/api/v1/transactions/demo-1/review", json={"decision": "note", "notes": "ok"}).status_code
        == 200
    )


def test_config_endpoint_outside_demo(client) -> None:
    cfg = client.get("/api/v1/config").json()
    assert cfg["demo_mode"] is False and cfg["policy_changes_allowed"] is True
