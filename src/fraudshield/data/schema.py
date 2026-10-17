"""Column contracts for the ULB credit-card dataset (mlg-ulb/creditcardfraud)."""

from __future__ import annotations

V_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
LABEL_COLUMN = "Class"
RAW_FEATURE_COLUMNS: tuple[str, ...] = (TIME_COLUMN, *V_COLUMNS, AMOUNT_COLUMN)
ALL_COLUMNS: tuple[str, ...] = (*RAW_FEATURE_COLUMNS, LABEL_COLUMN)

# Published characteristics of the canonical file; used to detect a wrong or corrupted download.
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
