"""IEEE-CIS Fraud Detection (Vesta) — second, independent benchmark.

Access: the data is distributed under Kaggle competition rules. You must accept them on
https://www.kaggle.com/competitions/ieee-fraud-detection/rules with your own account, create an API
token (kaggle.json) and run::

    kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-cis
    unzip data/raw/ieee-cis/ieee-fraud-detection.zip -d data/raw/ieee-cis

Only ``train_transaction.csv`` and ``train_identity.csv`` are used (the Kaggle test set has no
labels). Results are reported separately from the ULB credit-card benchmark: the datasets differ
in population, label definition and fraud rate, so their metrics must never be averaged.

Why this dataset matters here: unlike the ULB data (anonymised PCA components only), it carries
entity identifiers — card fields (card1-card6), billing address (addr1/addr2), e-mail domains and
device information — from which *relationships* between transactions can be derived honestly.

Leakage controls
* Out-of-time split on ``TransactionDT`` (seconds from an undisclosed reference).
* Categorical encodings are fit on the training period only; unseen categories map to -1.
* Entity/graph features use only *earlier* transactions (expanding counts in time order), so a
  transaction never sees its own label, future transactions or future entity activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "isFraud"
TIME = "TransactionDT"
ENTITY_KEYS: dict[str, list[str]] = {
    "card": ["card1", "card2", "card3", "card5"],
    "card_addr": ["card1", "addr1"],
    "card_email": ["card1", "P_emaildomain"],
    "device": ["DeviceInfo"],
    "email": ["P_emaildomain"],
}


def load_ieee_cis(root: str | Path, nrows: int | None = None) -> pd.DataFrame:
    root = Path(root)
    tx_path, id_path = root / "train_transaction.csv", root / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Accept the competition rules on Kaggle and download the data "
            "(see fraudshield/data/ieee_cis.py docstring or docs/ML_EVALUATION.md)."
        )
    tx = pd.read_csv(tx_path, nrows=nrows)
    if id_path.exists():
        tx = tx.merge(pd.read_csv(id_path), on="TransactionID", how="left")
    return tx.sort_values(TIME, kind="stable").reset_index(drop=True)


def entity_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(["<missing>"] * len(df), index=df.index)
    key = df[present[0]].astype("string").fillna("<na>")
    for c in present[1:]:
        key = key + "|" + df[c].astype("string").fillna("<na>")
    return key.astype(object)


def add_entity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Past-only relationship features (requires ``df`` sorted by time).

    For each entity key: number of earlier transactions with the same key, seconds since the
    previous one, and — for devices and e-mail domains — the number of distinct cards seen
    earlier (shared-device / shared-identity signal). Labels are never used.
    """
    out = df.copy()
    names: list[str] = []
    card = entity_key(out, ENTITY_KEYS["card"])
    for name, cols in ENTITY_KEYS.items():
        key = entity_key(out, cols)
        missing = (
            out[[c for c in cols if c in out.columns]].isna().any(axis=1)
            if any(c in out for c in cols)
            else True
        )
        prior = key.groupby(key).cumcount().astype(float)
        last_t = out.groupby(key)[TIME].shift(1)
        out[f"g_{name}_prior_txns"] = np.where(missing, np.nan, prior)
        out[f"g_{name}_secs_since_prev"] = np.where(missing, np.nan, out[TIME] - last_t)
        names += [f"g_{name}_prior_txns", f"g_{name}_secs_since_prev"]
        if name in ("device", "email"):
            pair = key + "#" + card
            first_pair = ~pair.duplicated(keep="first")
            distinct_before = first_pair.groupby(key).cumsum() - first_pair.astype(int)
            out[f"g_{name}_distinct_cards_before"] = np.where(missing, np.nan, distinct_before.astype(float))
            names.append(f"g_{name}_distinct_cards_before")
    return out, names


@dataclass
class IEEEPrepared:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    graph_features: list[str]


def prepare(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2, graph: bool = True
) -> IEEEPrepared:
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    graph_names: list[str] = []
    if graph:
        df, graph_names = add_entity_features(df)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    drop = {LABEL, "TransactionID", TIME}
    feats = [c for c in df.columns if c not in drop]
    x = df[feats].copy()
    tr_idx = np.arange(a)
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):  # object, str (pandas 3) or category
            cats = pd.Index(x[c].iloc[tr_idx].dropna().unique())  # learned on train only
            x[c] = pd.Categorical(x[c], categories=cats).codes.astype(np.int32)  # unseen/NaN -> -1
    y = df[LABEL].to_numpy()
    return IEEEPrepared(x.iloc[:a], x.iloc[a:b], x.iloc[b:], y[:a], y[a:b], y[b:], feats, graph_names)


def synthetic_ieee_like(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Schema-shaped synthetic frame for tests only (random values, NOT real data)."""
    rng = np.random.default_rng(seed)
    cards = rng.integers(1000, 1100, n)
    fraud = (rng.random(n) < 0.04).astype(int)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            TIME: np.sort(rng.integers(86_400, 86_400 * 30, n)),
            LABEL: fraud,
            "TransactionAmt": np.round(rng.lognormal(4, 1, n), 2),
            "ProductCD": rng.choice(["W", "C", "H", "R", "S"], n),
            "card1": cards,
            "card2": rng.integers(100, 600, n).astype(float),
            "card3": 150.0,
            "card5": 226.0,
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": np.where(rng.random(n) < 0.1, np.nan, rng.integers(100, 500, n)),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "anonymous.com", None], n),
            "DeviceInfo": np.where(
                fraud == 1, rng.choice(["devA", "devB"], n), rng.choice([f"dev{i}" for i in range(300)], n)
            ),
            "C1": rng.poisson(2 + 3 * fraud),
            "D1": rng.integers(0, 500, n).astype(float),
        }
    )
