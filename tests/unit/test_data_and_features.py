from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.data.creditcard import (
    DataValidationError,
    chronological_split,
    load_creditcard,
    validate_frame,
)
from fraudshield.data.schema import V_COLUMNS
from fraudshield.data.synthetic import generate_creditcard_like
from fraudshield.features.engineering import FeatureBuilder


@pytest.fixture
def df() -> pd.DataFrame:
    return generate_creditcard_like(n=3000, fraud_rate=0.02, seed=1)


def test_chronological_split_is_ordered_and_disjoint(df: pd.DataFrame) -> None:
    df = pd.concat([df, df.iloc[[100, 100, 100]]]).sort_values("Time", kind="stable")  # repeated timestamps
    s = chronological_split(df, 0.6, 0.2)
    assert s.train["Time"].max() < s.validation["Time"].min()
    assert s.validation["Time"].max() < s.test["Time"].min()
    assert len(s.train) + len(s.validation) + len(s.test) == len(df)
    assert abs(len(s.train) / len(df) - 0.6) < 0.01


def test_split_rejects_bad_fractions(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        chronological_split(df, 0.8, 0.3)


def test_load_drops_exact_duplicates(tmp_path, df: pd.DataFrame) -> None:
    dup = pd.concat([df, df.iloc[:25]])
    p = tmp_path / "d.csv"
    dup.to_csv(p, index=False)
    out, report = load_creditcard(p)
    assert report["exact_duplicates_removed"] == 25
    assert len(out) == len(df)
    assert out["Time"].is_monotonic_increasing


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["V3"]),
        lambda d: d.assign(Amount=-d["Amount"]),
        lambda d: d.assign(Class=2),
        lambda d: d.assign(V1=np.nan),
    ],
)
def test_validation_rejects_bad_frames(df: pd.DataFrame, mutate) -> None:
    with pytest.raises(DataValidationError):
        validate_frame(mutate(df.copy()))


def test_single_row_and_batch_features_are_identical(df: pd.DataFrame) -> None:
    """Regression test for the legacy train/serve skew (np.std vs pandas std, re-engineering)."""
    fb = FeatureBuilder().fit(df)
    batch = fb.transform(df)
    for i in (0, 17, 999):
        row = fb.transform(df.iloc[[i]])
        as_dict = fb.transform({c: np.array([df.iloc[i][c]]) for c in fb.raw_columns})
        np.testing.assert_array_equal(row[0], batch[i])
        np.testing.assert_array_equal(as_dict[0], batch[i])


def test_transform_does_not_depend_on_fitted_statistics(df: pd.DataFrame) -> None:
    a = FeatureBuilder().fit(df.iloc[:100]).transform(df)
    b = FeatureBuilder().fit(df.iloc[100:]).transform(df)
    np.testing.assert_array_equal(a, b)  # no statistic learned in fit() leaks into transform()


def test_raw_time_is_not_a_model_feature(df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    assert "Time" not in fb.feature_names
    shifted = df.assign(Time=df["Time"] + 86_400)  # exactly one day later
    np.testing.assert_allclose(fb.transform(df), fb.transform(shifted), atol=1e-9)


def test_engineered_values(df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    x = fb.transform(df.iloc[[0]])[0]
    names = fb.feature_names
    v = df.iloc[0][list(V_COLUMNS)].to_numpy(dtype=float)
    assert x[names.index("v_l2_norm")] == pytest.approx(np.linalg.norm(v))
    assert x[names.index("v_std")] == pytest.approx(v.std())
    assert x[names.index("amount_log1p")] == pytest.approx(np.log1p(df.iloc[0]["Amount"]))


def test_profile_percentile_is_monotone(df: pd.DataFrame) -> None:
    fb = FeatureBuilder().fit(df)
    lo, mid, hi = (fb.percentile("Amount", q) for q in (0.0, df["Amount"].median(), df["Amount"].max()))
    assert lo is not None and mid is not None and hi is not None
    assert lo <= mid <= hi
    assert 40 <= mid <= 60
    assert FeatureBuilder.from_dict(fb.to_dict()).percentile("Amount", 10.0) == fb.percentile("Amount", 10.0)
