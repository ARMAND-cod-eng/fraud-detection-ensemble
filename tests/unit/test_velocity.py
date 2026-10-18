"""Velocity features: hand-computed values, and proof that the online store equals the offline engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraudshield.features.velocity import (
    DAY,
    HOUR,
    VELOCITY_SPECS,
    OnlineFeatureStore,
    VelocitySpec,
    compute_offline,
)

CARD = ("card1",)
SPECS = (
    VelocitySpec("n_1h", CARD, HOUR, "count"),
    VelocitySpec("n_24h", CARD, DAY, "count"),
    VelocitySpec("amt_24h", CARD, DAY, "sum_amount"),
    VelocitySpec("ratio_24h", CARD, DAY, "amount_ratio_to_mean"),
    VelocitySpec("gap", CARD, DAY, "secs_since_prev"),
    VelocitySpec("emails_24h", CARD, DAY, "distinct", "email"),
    VelocitySpec("cards_on_email_1h", ("email",), HOUR, "distinct", "card1"),
)


def frame(rows: list[tuple[int, int | None, float, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["TransactionDT", "card1", "TransactionAmt", "email"])


def online(df: pd.DataFrame, specs=SPECS) -> pd.DataFrame:
    store = OnlineFeatureStore(specs)
    feats = [store.score_and_update(r) for r in df.to_dict("records")]
    return pd.DataFrame(feats, index=df.index)[[s.name for s in specs]]


def assert_parity(df: pd.DataFrame, specs=SPECS) -> None:
    off = compute_offline(df, specs)
    on = online(df, specs)
    np.testing.assert_allclose(on.to_numpy(float), off.to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True)


def test_hand_computed_values() -> None:
    df = frame(
        [
            (0, 1, 10.0, "a"),
            (600, 1, 30.0, "b"),  # same card 10 min later, new e-mail
            (4000, 1, 20.0, "a"),  # > 1 h after the first
            (4000, 2, 50.0, "a"),  # other card, same second
            (90_000, 1, 40.0, None),  # 25 h after the first, e-mail missing
        ]
    )
    off = compute_offline(df, SPECS)
    assert off["n_1h"].tolist() == [0, 1, 1, 0, 0]
    assert off["n_24h"].tolist() == [
        0,
        1,
        2,
        0,
        1,
    ]  # row 4: its window starts at t=3600, so only t=4000 counts
    assert off["amt_24h"].tolist() == [0, 10, 40, 0, 20]
    assert math.isnan(off["ratio_24h"].iloc[0])
    assert off["ratio_24h"].iloc[2] == pytest.approx(20 / 20)  # mean of earlier (10, 30) = 20
    assert off["gap"].iloc[1] == 600 and math.isnan(off["gap"].iloc[3])
    assert off["emails_24h"].tolist() == [0, 1, 2, 0, 1]  # the current (missing) e-mail is never counted
    # e-mail "a": card 1 at t=0 (outside 1 h at t=4000), so only card 1 at t=4000 precedes card 2
    assert off["cards_on_email_1h"].tolist()[:4] == [0, 0, 0, 1]
    assert math.isnan(off["cards_on_email_1h"].iloc[4])  # missing entity -> NaN
    assert_parity(df)


def test_missing_entity_is_nan_and_does_not_pollute_others() -> None:
    df = frame([(0, None, 5.0, "a"), (10, 1, 5.0, "a"), (20, None, 5.0, "a"), (30, 1, 5.0, "a")])
    off = compute_offline(df, SPECS)
    assert math.isnan(off["n_24h"].iloc[0]) and math.isnan(off["n_24h"].iloc[2])
    assert off["n_24h"].iloc[3] == 1  # rows without a card never count towards card 1
    assert_parity(df)


def test_unsorted_frame_rejected_and_bad_specs() -> None:
    with pytest.raises(ValueError):
        compute_offline(frame([(10, 1, 1.0, "a"), (5, 1, 1.0, "a")]), SPECS)
    with pytest.raises(ValueError):
        VelocitySpec("x", CARD, DAY, "distinct")
    with pytest.raises(ValueError):
        VelocitySpec("x", CARD, DAY, "median")


def test_future_events_do_not_change_past_features() -> None:
    rng = np.random.default_rng(0)
    n = 400
    df = frame(
        [
            (int(t), int(c), float(a), str(e))
            for t, c, a, e in zip(
                np.sort(rng.integers(0, 3 * DAY, n)),
                rng.integers(0, 15, n),
                rng.lognormal(3, 1, n),
                rng.choice(list("abcde"), n),
                strict=True,
            )
        ]
    )
    full = compute_offline(df, SPECS)
    prefix = compute_offline(df.iloc[:250], SPECS)
    pd.testing.assert_frame_equal(full.iloc[:250], prefix)


def test_warm_up_then_stream_equals_streaming_everything() -> None:
    rng = np.random.default_rng(1)
    n = 300
    df = frame(
        [
            (int(t), int(c), float(a), str(e))
            for t, c, a, e in zip(
                np.sort(rng.integers(0, 2 * DAY, n)),
                rng.integers(0, 10, n),
                rng.lognormal(3, 1, n),
                rng.choice(list("abc"), n),
                strict=True,
            )
        ]
    )
    records = df.to_dict("records")
    store = OnlineFeatureStore(SPECS)
    assert store.warm_up(records[:200]) == 200
    tail = pd.DataFrame([store.score_and_update(r) for r in records[200:]])[[s.name for s in SPECS]]
    expected = compute_offline(df, SPECS).iloc[200:].reset_index(drop=True)
    np.testing.assert_allclose(
        tail.to_numpy(float), expected.to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True
    )


def test_features_is_read_only() -> None:
    store = OnlineFeatureStore(SPECS)
    r = {"TransactionDT": 0, "card1": 1, "TransactionAmt": 5.0, "email": "a"}
    store.update(r)
    probe = {**r, "TransactionDT": 10}
    assert store.features(probe) == store.features(probe)  # reading twice changes nothing
    assert store.features(probe)["n_24h"] == 1 and store.events_seen == 1


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(
            st.integers(0, 4000),  # time increment (many ties and window edges)
            st.one_of(st.none(), st.integers(0, 4)),  # card, sometimes missing
            st.integers(0, 500_000).map(lambda cents: cents / 100),  # amount in whole cents
            st.one_of(st.none(), st.sampled_from(["a", "b", "c"])),  # e-mail, sometimes missing
        ),
        min_size=1,
        max_size=80,
    )
)
def test_online_equals_offline_on_random_streams(events) -> None:
    t, rows = 0, []
    for dt, card, amount, email in events:
        t += dt if dt % 3 else 0  # a third of events share the previous timestamp
        rows.append((t, card, amount, email))
    assert_parity(frame(rows))


def test_production_specs_parity_on_ieee_like_data() -> None:
    from fraudshield.data.ieee_cis import synthetic_ieee_like

    df = synthetic_ieee_like(n=1500, seed=4)
    df["DeviceInfo"] = df["DeviceInfo"].where(np.arange(len(df)) % 7 != 0)  # some missing devices
    assert_parity(df, VELOCITY_SPECS)
