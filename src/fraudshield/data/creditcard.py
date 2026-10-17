"""Loading, validation and leakage-safe splitting of the credit-card dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.data.schema import ALL_COLUMNS, LABEL_COLUMN, TIME_COLUMN


class DataValidationError(ValueError):
    """Raised when a dataset does not satisfy the expected contract."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def validate_frame(df: pd.DataFrame) -> None:
    """Schema and sanity checks that must pass before any training run."""
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    values = df[list(ALL_COLUMNS)].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise DataValidationError("dataset contains null values")
    if not np.isfinite(values).all():
        raise DataValidationError("dataset contains non-finite values")
    if not set(np.unique(df[LABEL_COLUMN])) <= {0, 1}:
        raise DataValidationError("label column must be binary 0/1")
    if (df["Amount"] < 0).any():
        raise DataValidationError("negative transaction amounts")
    if df[LABEL_COLUMN].sum() == 0:
        raise DataValidationError("dataset contains no positive (fraud) examples")


def load_creditcard(path: str | Path, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, validate it and (optionally) drop exact duplicate rows.

    Exact duplicates (1,081 rows in the canonical file) can land on both sides of a random
    split and inflate test metrics, so they are removed before splitting.
    Returns the frame (stable-sorted by time, index reset) and a data report for the model card.
    """
    path = Path(path)
    df = pd.read_csv(path)
    validate_frame(df)
    report: dict = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows_raw": len(df),
        "frauds_raw": int(df[LABEL_COLUMN].sum()),
        "time_span_hours": round(float(df[TIME_COLUMN].max() - df[TIME_COLUMN].min()) / 3600, 2),
    }
    if drop_duplicates:
        dup = df.duplicated(keep="first")
        report["exact_duplicates_removed"] = int(dup.sum())
        report["fraud_duplicates_removed"] = int(df.loc[dup, LABEL_COLUMN].sum())
        df = df.loc[~dup]
    df = df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    report["rows"] = len(df)
    report["frauds"] = int(df[LABEL_COLUMN].sum())
    return df, report


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str

    def summary(self) -> dict:
        out: dict = {"strategy": self.strategy}
        for name in ("train", "validation", "test"):
            part: pd.DataFrame = getattr(self, name)
            out[name] = {
                "rows": len(part),
                "frauds": int(part[LABEL_COLUMN].sum()),
                "fraud_rate": round(float(part[LABEL_COLUMN].mean()), 6),
                "time_start_s": float(part[TIME_COLUMN].min()),
                "time_end_s": float(part[TIME_COLUMN].max()),
            }
        return out


def chronological_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> Split:
    """Out-of-time split: train on the earliest transactions, validate and test on later ones.

    Boundaries are placed on distinct timestamps so that rows sharing a ``Time`` value never
    straddle two partitions. Every statistic, threshold and calibration map must be learned on
    ``train``/``validation`` only; ``test`` is reserved for the final report.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must satisfy 0 < train, val and train + val < 1")
    df = df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    times = df[TIME_COLUMN].to_numpy()
    n = len(df)

    def boundary(frac: float) -> int:
        i = int(n * frac)
        # advance to the first row whose timestamp differs from the previous row
        while 0 < i < n and times[i] == times[i - 1]:
            i += 1
        return i

    a, b = boundary(train_frac), boundary(train_frac + val_frac)
    return Split(
        train=df.iloc[:a].reset_index(drop=True),
        validation=df.iloc[a:b].reset_index(drop=True),
        test=df.iloc[b:].reset_index(drop=True),
        strategy=f"chronological {train_frac:.0%}/{val_frac:.0%}/{1 - train_frac - val_frac:.0%}",
    )


def stratified_random_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 42
) -> Split:
    """Random stratified split, provided only for comparison with the legacy protocol."""
    from sklearn.model_selection import train_test_split

    rest_frac = 1 - train_frac
    tr, rest = train_test_split(df, test_size=rest_frac, random_state=seed, stratify=df[LABEL_COLUMN])
    va, te = train_test_split(
        rest,
        test_size=(1 - train_frac - val_frac) / rest_frac,
        random_state=seed,
        stratify=rest[LABEL_COLUMN],
    )
    return Split(
        train=tr.reset_index(drop=True),
        validation=va.reset_index(drop=True),
        test=te.reset_index(drop=True),
        strategy=f"stratified-random seed={seed}",
    )
