"""Serving-profile plumbing: settings defaults, NaN-aware drift, store migration, payload hashing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from fraudshield.config import Settings
from fraudshield.monitoring.drift import build_reference, drift_report
from fraudshield.profiles import get_profile
from fraudshield.profiles.base import entity_hash, json_value
from fraudshield.profiles.creditcard import CreditCardProfile
from fraudshield.profiles.ieee_serving import IEEEProfile
from fraudshield.storage.store import PredictionStore


def test_profile_defaults_only_fill_unset_paths() -> None:
    s = Settings(profile="ieee", db_path=Path("custom.db"))
    assert s.artifacts_dir == Path("artifacts/ieee") and s.db_path == Path("custom.db")
    assert s.warmup_file == Path("artifacts/ieee/replay/ieee_warmup.csv.gz")
    assert isinstance(get_profile(s), IEEEProfile)
    c = Settings()
    assert c.profile == "creditcard" and c.artifacts_dir == Path("artifacts") and c.warmup_file is None
    assert isinstance(get_profile(c), CreditCardProfile)


def test_payload_hash_is_stable_and_type_tolerant() -> None:
    p = IEEEProfile(None, None)
    a = {"TransactionDT": 10, "TransactionAmt": 5, "card1": 13926, "P_emaildomain": "gmail.com"}
    b = {"TransactionDT": 10.0, "TransactionAmt": 5.0, "card1": 13926.0, "P_emaildomain": "gmail.com"}
    assert p.payload_hash(a) == p.payload_hash(b)  # JSON ints vs CSV floats
    assert p.payload_hash(a) != p.payload_hash({**a, "P_emaildomain": "yahoo.com"})
    assert p.payload_hash({**a, "dist1": float("nan")}) == p.payload_hash({**a, "dist1": None})
    assert entity_hash("card", (1, 2.0)) == entity_hash("card", (1.0, 2))
    assert json_value(np.float32(1.5)) == 1.5 and json_value(float("nan")) is None


def test_creditcard_hash_unchanged_from_v1() -> None:
    """Existing databases keep working: the credit-card hash format did not change."""
    import hashlib
    import json

    p = CreditCardProfile(None)
    rec = {c: float(i) / 7 for i, c in enumerate(p.raw_fields)}
    v1 = hashlib.sha256(json.dumps([repr(float(rec[c])) for c in p.raw_fields]).encode()).hexdigest()
    assert p.payload_hash(rec) == v1


def test_drift_has_a_missing_bin() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(5000, 1))
    train[:500, 0] = np.nan  # 10% missing at training time
    ref = build_reference(train, ["x"], rng.random(5000))
    assert ref["features"]["x"]["missing_bin"] and np.isfinite(ref["features"]["x"]["edges"]).all()
    same = rng.normal(size=(2000, 1))
    same[:200, 0] = np.nan
    more_missing = rng.normal(size=(2000, 1))
    more_missing[:1200, 0] = np.nan  # same values, but 60% missing
    ok = drift_report(ref, same, ["x"])["features"][0]
    shifted = drift_report(ref, more_missing, ["x"])["features"][0]
    assert ok["status"] == "ok" and shifted["status"] == "alert"


def test_drift_reads_old_references_without_missing_bin() -> None:
    rng = np.random.default_rng(1)
    edges = np.quantile(rng.normal(size=1000), np.linspace(0, 1, 11)[1:-1])
    old = {"features": {"x": {"edges": edges.tolist(), "expected": [0.1] * 10}}}
    rep = drift_report(old, rng.normal(size=(1000, 1)), ["x"])
    assert rep["features"][0]["status"] == "ok"


def test_store_migrates_old_databases(tmp_path) -> None:
    from fraudshield.storage.store import SCHEMA

    v1_schema = SCHEMA.replace(",\n    top_reasons       TEXT", "")
    assert v1_schema != SCHEMA
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(v1_schema.split("-- Linked activity")[0])  # v1: no top_reasons, no entity_links
    con.commit()
    con.close()
    store = PredictionStore(path)
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(predictions)").fetchall()}
    assert "top_reasons" in cols
    tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "entity_links" in tables
    store.close()
