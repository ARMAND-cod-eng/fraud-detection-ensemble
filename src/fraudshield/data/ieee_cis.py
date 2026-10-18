"""IEEE-CIS Fraud Detection (Vesta) — second, independent benchmark.

Access: the data is distributed under Kaggle competition rules. Accept them on
https://www.kaggle.com/competitions/ieee-fraud-detection/rules with your own account, download the
data (web "Download All", or ``kaggle competitions download -c ieee-fraud-detection``) and extract
``train_transaction.csv`` and ``train_identity.csv`` into ``data/raw/ieee-cis/``.

Only the two training files are used (the Kaggle test set has no labels). Results are reported
separately from the ULB credit-card benchmark: the datasets differ in population, label definition
and fraud rate, so their metrics must never be averaged.

Why this dataset matters here: unlike the ULB data (anonymised PCA components only), it carries
entity identifiers — card fields (card1-card6), billing address (addr1/addr2), e-mail domains and
device information — from which *relationships* between transactions can be derived honestly.

Leakage controls
* Out-of-time split on ``TransactionDT`` (seconds from an undisclosed reference).
* Categorical encodings are fit on the training period only; unseen categories map to -1.
* Entity features use only *earlier* transactions, so a transaction never sees its own label,
  future transactions or future entity activity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.features.velocity import window_counts, window_distinct

LABEL = "isFraud"
TIME = "TransactionDT"
ENTITY_KEYS: dict[str, list[str]] = {
    "card": ["card1", "card2", "card3", "card5"],
    "card_addr": ["card1", "addr1"],
    "card_email": ["card1", "P_emaildomain"],
    "device": ["DeviceInfo"],
    "email": ["P_emaildomain"],
}
WINDOWS_S: dict[str, int] = {"24h": 86_400, "7d": 7 * 86_400}
ENTITY_MODES = ("cumulative", "windowed")

# Text columns of train_transaction.csv; every other column is numeric.
TX_TEXT_COLUMNS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 and text -> category (about half the memory of pandas defaults)."""
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


def load_ieee_cis(root: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load train_transaction (+ train_identity) with compact dtypes, sorted by time."""
    root = Path(root)
    tx_path, id_path = root / "train_transaction.csv", root / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Accept the competition rules on Kaggle and download the data "
            "(see fraudshield/data/ieee_cis.py docstring or docs/ML_EVALUATION.md)."
        )
    header = pd.read_csv(tx_path, nrows=0).columns
    dtypes: dict[str, str] = {c: "float32" for c in header}
    dtypes.update({c: "category" for c in TX_TEXT_COLUMNS if c in header})
    dtypes.update({"TransactionID": "int64", TIME: "int64", LABEL: "int8"})
    tx = pd.read_csv(tx_path, nrows=nrows, dtype=dtypes)
    if id_path.exists():
        # Add identity columns one by one instead of merge() to avoid copying the 590k x 394 frame.
        ident = _compact(pd.read_csv(id_path)).set_index("TransactionID").reindex(tx["TransactionID"])
        with warnings.catch_warnings():  # fragmentation is harmless here; a consolidating copy is not
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for c in ident.columns:
                tx[c] = ident[c].to_numpy()
    if not tx[TIME].is_monotonic_increasing:
        tx = tx.sort_values(TIME, kind="stable").reset_index(drop=True)
    return tx


def load_contract(root: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load only the authorisation-time field contract (v2), sorted by time."""
    from fraudshield.profiles.ieee import IDENTITY_CONTRACT_FIELDS, TX_CONTRACT_FIELDS

    root = Path(root)
    tx_path, id_path = root / "train_transaction.csv", root / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"{tx_path} not found (see the docstring of this module)")
    cols = ["TransactionID", TIME, LABEL, *TX_CONTRACT_FIELDS]
    text = {c: "object" for c in TX_TEXT_COLUMNS if c in cols}
    tx = pd.read_csv(tx_path, usecols=cols, nrows=nrows, dtype=text)
    if id_path.exists():
        ident = pd.read_csv(id_path, usecols=["TransactionID", *IDENTITY_CONTRACT_FIELDS], dtype="object")
        ident["TransactionID"] = ident["TransactionID"].astype("int64")
        ident = ident.set_index("TransactionID").reindex(tx["TransactionID"])
        for c in IDENTITY_CONTRACT_FIELDS:
            tx[c] = ident[c].to_numpy()
    else:
        for c in IDENTITY_CONTRACT_FIELDS:
            tx[c] = None
    return tx.sort_values(TIME, kind="stable").reset_index(drop=True)


def entity_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(["<missing>"] * len(df), index=df.index)
    key = df[present[0]].astype("string").fillna("<na>")
    for c in present[1:]:
        key = key + "|" + df[c].astype("string").fillna("<na>")
    return key.astype(object)


def add_entity_features(
    df: pd.DataFrame, copy: bool = True, mode: str = "cumulative"
) -> tuple[pd.DataFrame, list[str]]:
    """Past-only relationship features (requires ``df`` sorted by time). Labels are never used.

    ``mode="cumulative"``: per entity, transactions since the start of the data, seconds since the
    previous one, and (devices / e-mail domains) distinct cards seen so far. These counts only grow
    with time, so later periods fall outside the training range; on IEEE-CIS they *hurt*
    out-of-time PR-AUC (docs/ML_EVALUATION.md).
    ``mode="windowed"``: the same signals over trailing windows (24 h and 7 days), which stay
    comparable across periods.
    ``copy=False`` adds the columns in place (used for the 590k-row dataset).
    """
    if mode not in ENTITY_MODES:
        raise ValueError(f"mode must be one of {ENTITY_MODES}")
    out = df.copy() if copy else df
    with warnings.catch_warnings():  # adding columns to a wide frame; fragmentation is harmless
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _cumulative(out) if mode == "cumulative" else _windowed(out)


def _missing(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    present = [c for c in cols if c in df.columns]
    if not present:
        return np.ones(len(df), dtype=bool)
    return df[present].isna().any(axis=1).to_numpy()


def _cumulative(out: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    names: list[str] = []
    card = entity_key(out, ENTITY_KEYS["card"])
    for name, cols in ENTITY_KEYS.items():
        key = entity_key(out, cols)
        missing = _missing(out, cols)
        prior = key.groupby(key).cumcount().astype(float)
        last_t = out.groupby(key)[TIME].shift(1)
        out[f"g_{name}_prior_txns"] = np.where(missing, np.nan, prior).astype(np.float32)
        out[f"g_{name}_secs_since_prev"] = np.where(missing, np.nan, out[TIME] - last_t).astype(np.float32)
        names += [f"g_{name}_prior_txns", f"g_{name}_secs_since_prev"]
        if name in ("device", "email"):
            pair = key + "#" + card
            first_pair = ~pair.duplicated(keep="first")
            distinct_before = first_pair.groupby(key).cumsum() - first_pair.astype(int)
            col = f"g_{name}_distinct_cards_before"
            out[col] = np.where(missing, np.nan, distinct_before.astype(float)).astype(np.float32)
            names.append(col)
    return out, names


def _windowed(out: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    names: list[str] = []
    t = out[TIME].to_numpy()
    card_codes = pd.factorize(entity_key(out, ENTITY_KEYS["card"]))[0]
    for name, cols in ENTITY_KEYS.items():
        key = entity_key(out, cols)
        codes = pd.factorize(key)[0]
        missing = _missing(out, cols)
        for label, w in WINDOWS_S.items():
            col = f"w_{name}_txns_{label}"
            out[col] = np.where(missing, np.nan, window_counts(codes, t, w)).astype(np.float32)
            names.append(col)
        last_t = out.groupby(key)[TIME].shift(1)
        out[f"w_{name}_secs_since_prev"] = np.where(missing, np.nan, out[TIME] - last_t).astype(np.float32)
        names.append(f"w_{name}_secs_since_prev")
        if name in ("device", "email"):
            col = f"w_{name}_distinct_cards_7d"
            distinct = window_distinct(codes, card_codes, t, WINDOWS_S["7d"])
            out[col] = np.where(missing, np.nan, distinct).astype(np.float32)
            names.append(col)
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
    entity_features: dict[str, list[str]]

    @property
    def graph_features(self) -> list[str]:
        return [f for names in self.entity_features.values() for f in names]

    def columns(self, mode: str | None) -> list[str]:
        """Base columns plus the entity features of ``mode`` (None = base columns only)."""
        others = {f for m, names in self.entity_features.items() if m != mode for f in names}
        return [c for c in self.feature_names if c not in others]


def prepare(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    graph: bool = True,
    modes: tuple[str, ...] = ("cumulative",),
) -> IEEEPrepared:
    if not df[TIME].is_monotonic_increasing:
        df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    entity: dict[str, list[str]] = {}
    if graph:
        for mode in modes:
            df, entity[mode] = add_entity_features(df, copy=False, mode=mode)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    drop = {LABEL, "TransactionID", TIME}
    feats = [c for c in df.columns if c not in drop]
    x = df[feats]  # copy-on-write: only the re-encoded columns are materialised
    tr_idx = np.arange(a)
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):  # object, str (pandas 3) or category
            cats = pd.Index(x[c].iloc[tr_idx].dropna().unique())  # learned on train only
            vals = x[c].astype(object)
            seen = vals.where(vals.isin(cats))  # unseen categories -> missing
            x[c] = pd.Categorical(seen, categories=cats).codes.astype(np.int32)  # missing -> -1
    y = df[LABEL].to_numpy()
    return IEEEPrepared(x.iloc[:a], x.iloc[a:b], x.iloc[b:], y[:a], y[a:b], y[b:], feats, entity)


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
            "addr2": np.where(rng.random(n) < 0.1, np.nan, 87.0),
            "dist1": np.where(rng.random(n) < 0.5, np.nan, rng.integers(0, 300, n).astype(float)),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "anonymous.com", None], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None, None], n),
            **{f"M{i}": rng.choice(["T", "F", None], n) for i in range(1, 10)},
            "DeviceType": rng.choice(["desktop", "mobile", None], n),
            "DeviceInfo": np.where(
                fraud == 1, rng.choice(["devA", "devB"], n), rng.choice([f"dev{i}" for i in range(300)], n)
            ),
            "C1": rng.poisson(2 + 3 * fraud),
            "D1": rng.integers(0, 500, n).astype(float),
        }
    )
