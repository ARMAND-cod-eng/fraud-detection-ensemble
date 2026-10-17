"""The IEEE-CIS pipeline is tested on a synthetic, schema-shaped frame (the real data needs Kaggle terms)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.data.ieee_cis import add_entity_features, prepare, synthetic_ieee_like


def test_entity_features_use_only_the_past() -> None:
    df = pd.DataFrame(
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
    out, names = add_entity_features(df)
    assert out["g_card_prior_txns"].tolist() == [0, 1, 0, 2, 1]
    assert out["g_card_secs_since_prev"].tolist()[1] == 10
    assert np.isnan(out["g_card_addr_prior_txns"].iloc[3])  # missing addr -> no entity
    # device d1 is used by card 1 (t=10,20) and card 2 (t=30, 50): distinct earlier cards
    assert out["g_device_distinct_cards_before"].tolist() == [0, 1, 1, 0, 2]
    # changing a later label must not change any earlier feature
    df2 = df.copy()
    df2.loc[4, "isFraud"] = 0
    out2, _ = add_entity_features(df2)
    pd.testing.assert_frame_equal(out[names], out2[names])


def test_prepare_encodes_categories_on_train_only() -> None:
    df = synthetic_ieee_like(n=1000, seed=1)
    df.loc[df.index[-5:], "ProductCD"] = "UNSEEN"
    p = prepare(df, graph=True)
    assert (p.x_test["ProductCD"].iloc[-5:] == -1).all()
    assert p.x_train.index.max() < p.x_val.index.min() < p.x_test.index.min()
    assert set(p.graph_features) <= set(p.feature_names)
    assert len(p.y_train) + len(p.y_val) + len(p.y_test) == len(df)
