"""Credit-card ("creditcard") serving profile: the ULB dataset, Time + V1..V28 + Amount.

The data has no card or customer identifiers, so there are no online entity features and no
linked-activity view; every transaction is scored on its own fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.data.schema import AMOUNT_COLUMN, LABEL_COLUMN, RAW_FEATURE_COLUMNS, TIME_COLUMN, V_COLUMNS
from fraudshield.data.synthetic import generate_creditcard_like
from fraudshield.features.engineering import FEATURE_DESCRIPTIONS
from fraudshield.profiles.base import ReplayItem, ServingProfile

SCENARIOS: dict[str, str] = {
    "historical": "Held-out test window, chronological, natural fraud rate (~0.13%)",
    "enriched": "Held-out transactions resampled to up to 5% real frauds, each at most once (inflated alert rate)",
    "synthetic_drift": "SYNTHETIC: held-out rows with injected Amount/component shift (drift demo)",
}
ATTRIBUTION = (
    "Data: Credit Card Fraud Detection, Machine Learning Group - ULB and Worldline "
    "(Open Database License 1.0 / Database Contents License 1.0)."
)


class CreditCardProfile(ServingProfile):
    name = "creditcard"
    preprocessor_kind = "creditcard"
    raw_fields = RAW_FEATURE_COLUMNS
    time_field = TIME_COLUMN
    amount_field = AMOUNT_COLUMN
    label_field = LABEL_COLUMN
    currency = "EUR"
    data_attribution = ATTRIBUTION
    scenarios = SCENARIOS

    def __init__(self, replay_file: Path | None) -> None:
        self.replay_file = replay_file
        self._frame: pd.DataFrame | None = None
        self._source_name = ""

    def quality_warnings(self, model: Any, base_row: np.ndarray, record: Mapping[str, Any]) -> list[str]:
        warnings = super().quality_warnings(model, base_row, record)
        if record[AMOUNT_COLUMN] == 0:
            warnings.append("Amount is 0 (zero-value authorisation)")
        if record[TIME_COLUMN] > 172_800:
            warnings.append(
                "Time beyond the 48 h capture window; hour-of-day encoding assumes Time=0 at midnight"
            )
        return warnings[:10]

    def feature_description(self, feature: str) -> str:
        return FEATURE_DESCRIPTIONS.get(feature, feature)

    # ------------------------------------------------------------------ data sources
    def _source(self) -> tuple[pd.DataFrame, str]:
        if self._frame is None:
            if self.replay_file is not None and Path(self.replay_file).exists():
                self._frame = pd.read_csv(self.replay_file)
                self._source_name = f"held-out test window ({Path(self.replay_file).name})"
            else:
                self._frame = generate_creditcard_like(n=20_000, fraud_rate=0.01, seed=7)
                self._source_name = "SYNTHETIC fallback (replay file not found)"
        return self._frame, self._source_name

    def replay_items(self, scenario: str, count: int, seed: int) -> tuple[list[ReplayItem], list[str]]:
        df, source_name = self._source()
        rng = np.random.default_rng(seed)
        notes = [f"source: {source_name}"]
        synthetic_source = source_name.startswith("SYNTHETIC")
        if scenario == "historical":
            start = int(rng.integers(0, max(1, len(df) - count))) if len(df) > count else 0
            sample = df.iloc[start : start + count]
            notes.append(f"contiguous chronological slice starting at row {start}")
        elif scenario == "enriched":
            fraud = df[df[LABEL_COLUMN] == 1]
            legit = df[df[LABEL_COLUMN] == 0]
            # Each real fraud appears at most once: long runs get a lower fraud share instead of copies.
            n_fraud = max(1, min(round(count * 0.05), len(fraud)))
            idx_f = rng.choice(len(fraud), size=n_fraud, replace=False)
            idx_l = rng.choice(len(legit), size=min(count - n_fraud, len(legit)), replace=False)
            sample = pd.concat([fraud.iloc[idx_f], legit.iloc[idx_l]]).sort_values(TIME_COLUMN, kind="stable")
            notes.append(
                f"{n_fraud} distinct real frauds among {len(sample)} rows ({n_fraud / len(sample):.1%})"
            )
        elif scenario == "synthetic_drift":
            sample = df.iloc[rng.choice(len(df), size=min(count, len(df)), replace=count > len(df))].copy()
            sample[AMOUNT_COLUMN] = (sample[AMOUNT_COLUMN] * rng.uniform(2.0, 3.5, len(sample))).round(2)
            for c in V_COLUMNS[:6]:
                sample[c] = sample[c] + 1.5 + rng.normal(0, 0.3, len(sample))
            notes.append("Amount x2-3.5 and V1-V6 shifted by +1.5: SYNTHETIC rows, no ground truth")
        else:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(self.scenarios)}")
        has_labels = scenario != "synthetic_drift" and not synthetic_source
        synthetic = scenario == "synthetic_drift" or synthetic_source
        items: list[ReplayItem] = [
            (rec, int(label) if has_labels else None, synthetic)
            for rec, label in zip(
                sample[list(RAW_FEATURE_COLUMNS)].to_dict("records"),
                sample[LABEL_COLUMN].tolist(),
                strict=True,
            )
        ]
        return items, notes

    def held_out(self) -> tuple[Any, np.ndarray, np.ndarray, float]:
        if self.replay_file is None or not Path(self.replay_file).exists():
            raise FileNotFoundError("held-out replay file not found; run training first")
        df = pd.read_csv(self.replay_file)
        cols = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_FEATURE_COLUMNS}
        hours = float(df[TIME_COLUMN].max() - df[TIME_COLUMN].min()) / 3600
        return cols, df[LABEL_COLUMN].to_numpy(), df[AMOUNT_COLUMN].to_numpy(), hours
