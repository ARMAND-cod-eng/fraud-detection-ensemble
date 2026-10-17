"""The IEEE-CIS pipeline is tested on hand-built and synthetic frames (the real data needs Kaggle terms)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.data.ieee_cis import (
    add_entity_features,
    prepare,
    synthetic_ieee_like,
    window_counts,
    window_distinct,
)

DAY = 86_400


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [10, 20, 30, 40, 50],
            "isFraud": [0, 1, 0, 0, 1],
            "card1": [1, 1, 2, 1, 2],
            "card2": 1.0,
            "card3": 1.0,
            "card5": 1.0,
            "addr1": [5, 5, 5, np.nan, 5],
            "P_emaildomain": ["a", "a", "b", "a", "b"],
            "DeviceInfo": ["d1", "d1", "d1", "d2", "d1"],
        }
    )


def test_cumulative_entity_features_use_only_the_past() -> None:
    df = _frame()
    out, names = add_entity_features(df)
    assert out["g_card_prior_txns"].tolist() == [0, 1, 0, 2, 1]
    assert out["g_card_secs_since_prev"].tolist()[1] == 10
    assert np.isnan(out["g_card_addr_prior_txns"].iloc[3])  # missing addr -> no entity
    # device d1 is used by card 1 (t=10,20) and card 2 (t=30, 50): distinct earlier cards
    assert out["g_device_distinct_cards_before"].tolist() == [0, 1, 1, 0, 2]
    df2 = df.copy()
    df2.loc[4, "isFraud"] = 0  # changing a label must not change any feature
    out2, _ = add_entity_features(df2)
    pd.testing.assert_frame_equal(out[names], out2[names])


def test_window_counts_hand_computed() -> None:
    t = np.array([0, 3_600, 90_000, 100_000, 700_000])
    same = np.zeros(5, dtype=int)
    assert window_counts(same, t, DAY).tolist() == [0, 1, 1, 1, 0]
    assert window_counts(same, t, 7 * DAY).tolist() == [0, 1, 2, 3, 1]
    two = np.array([0, 1, 0, 1, 0])  # interleaved entities never count each other
    assert window_counts(two, t, 7 * DAY).tolist() == [0, 0, 1, 1, 0]  # row 4 is >7 days after row 2
    ties = window_counts(np.zeros(3, dtype=int), np.array([5, 5, 5]), DAY)
    assert ties.tolist() == [0, 1, 2]  # same second: only rows earlier in the stream count


def test_window_distinct_hand_computed() -> None:
    t = np.array([0, 10, 20, 8 * DAY, 8 * DAY + 5])
    device = np.array([0, 0, 0, 0, 1])
    card = np.array([7, 8, 7, 9, 7])
    # before row 2: cards {7, 8}; row 3 is 8 days later so the window has expired
    assert window_distinct(device, card, t, 7 * DAY).tolist() == [0, 1, 2, 0, 0]


def test_windowed_features_are_past_only_and_stationary() -> None:
    df = synthetic_ieee_like(n=3000, seed=2)
    out, names = add_entity_features(df, mode="windowed")
    assert all(n.startswith("w_") for n in names) and len(names) == 17
    # truncating the future must not change features of earlier rows
    early, _ = add_entity_features(df.iloc[:1500].copy(), mode="windowed")
    pd.testing.assert_frame_equal(out[names].iloc[:1500].reset_index(drop=True), early[names])
    # counts are bounded by window activity, not by elapsed time (unlike cumulative counts)
    cum, _ = add_entity_features(df, mode="cumulative")
    late = slice(2500, 3000)
    assert out["w_card_txns_7d"].iloc[late].mean() < cum["g_card_prior_txns"].iloc[late].mean()
    with pytest.raises(ValueError):
        add_entity_features(df, mode="bogus")


def test_prepare_encodes_categories_on_train_only_and_selects_columns() -> None:
    df = synthetic_ieee_like(n=1000, seed=1)
    df.loc[df.index[-5:], "ProductCD"] = "UNSEEN"
    p = prepare(df, graph=True, modes=("cumulative", "windowed"))
    assert (p.x_test["ProductCD"].iloc[-5:] == -1).all()
    assert p.x_train.index.max() < p.x_val.index.min() < p.x_test.index.min()
    base, cum, win = p.columns(None), p.columns("cumulative"), p.columns("windowed")
    assert not any(c.startswith(("g_", "w_")) for c in base)
    assert set(cum) - set(base) == set(p.entity_features["cumulative"])
    assert set(win) - set(base) == set(p.entity_features["windowed"])
    assert len(p.y_train) + len(p.y_val) + len(p.y_test) == len(df)
