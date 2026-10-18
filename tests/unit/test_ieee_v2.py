"""FraudShield v2 (IEEE-CIS profile) on synthetic, schema-shaped data: preprocessing parity and pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraudshield.data.ieee_cis import synthetic_ieee_like
from fraudshield.features.velocity import compute_offline
from fraudshield.models.artifact import ModelRegistry, load_model
from fraudshield.profiles.ieee import IEEEPreprocessor, reason_text

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    df = synthetic_ieee_like(n=3000, seed=5)
    return pd.concat([df, compute_offline(df)], axis=1)


def test_single_record_path_equals_batch_path(frame) -> None:
    """The serving fast path must produce exactly the training features (no train/serve skew)."""
    prep = IEEEPreprocessor(velocity=True).fit(frame.iloc[:2000])
    late = frame.iloc[2000:].copy()
    late.loc[late.index[:20], "P_emaildomain"] = "never-seen.example"  # unseen category
    late.loc[late.index[20:40], "TransactionAmt"] = np.nan  # missing numeric
    batch = prep.transform(late)
    for i, rec in enumerate(late.to_dict("records")[:300]):
        np.testing.assert_array_equal(prep.transform_record(rec)[0], batch[i])
    one = prep.transform({k: [v] for k, v in late.iloc[0].to_dict().items()})  # API-style mapping
    np.testing.assert_array_equal(one[0], batch[0])
    assert (batch[:20, prep.feature_names.index("P_emaildomain")] == -1).all()


def test_preprocessor_roundtrip(frame) -> None:
    prep = IEEEPreprocessor(velocity=False).fit(frame)
    clone = IEEEPreprocessor.from_dict(prep.to_dict())
    np.testing.assert_array_equal(clone.transform(frame), prep.transform(frame))
    assert prep.percentile("TransactionAmt", float(frame["TransactionAmt"].median())) == pytest.approx(
        50, abs=3
    )
    bad = prep.to_dict() | {"feature_names": ["x"]}
    with pytest.raises(ValueError):
        IEEEPreprocessor.from_dict(bad)


def test_reason_text() -> None:
    assert (
        reason_text("v_card_count_24h", 7.0, 1.0)
        == "7 earlier transactions on this card in the last 24 h (typical: 1)"
    )
    assert reason_text("v_card_count_24h", 1.0, 0.0).startswith("1 earlier transaction on this card")
    assert reason_text("v_device_distinct_cards_24h", 1.0, None) == (
        "1 different card used with this device model in 24 h"
    )
    assert reason_text("v_card_amount_ratio_7d", 3.2, 1.0).startswith("amount is 3.2x")
    assert reason_text("TransactionAmt", 10.0, None) is None
    assert reason_text("v_card_count_24h", float("nan"), 1.0) is None


def test_v2_pipeline_end_to_end(tmp_path, monkeypatch) -> None:
    from fraudshield.training.ieee_v2 import run_ieee_v2

    monkeypatch.setenv("FRAUDSHIELD_DISABLE_MLFLOW", "1")
    cfg = yaml.safe_load((ROOT / "configs" / "ieee_v2.yaml").read_text())
    out = run_ieee_v2(
        cfg,
        frame=synthetic_ieee_like(n=4000, seed=0),
        artifacts_root=tmp_path / "a",
        results_dir=tmp_path / "r",
        quick=True,
    )
    ev = out["evaluation"]
    assert ev["selected_model"] in {"raw_contract", "raw_plus_velocity"}
    assert ev["protocol"]["selection_log"] and ev["backtest"]["blocks"]
    assert ModelRegistry(tmp_path / "a").production() == out["version"]  # passed the gate
    model, meta = load_model(out["model_dir"])
    assert isinstance(model.feature_builder, IEEEPreprocessor) and meta["profile"] == "ieee"
    replay = pd.read_csv(tmp_path / "a" / "replay" / "ieee_test_window.csv.gz")
    warm = pd.read_csv(tmp_path / "a" / "replay" / "ieee_warmup.csv.gz")
    assert warm["TransactionDT"].max() < replay["TransactionDT"].min()  # warm-up strictly precedes replay
    assert "Monthly backtest" in (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
