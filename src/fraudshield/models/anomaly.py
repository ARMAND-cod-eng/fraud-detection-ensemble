"""Isolation Forest anomaly detector (unsupervised; never sees labels)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestDetector:
    n_estimators: int = 200
    max_samples: int = 256
    seed: int = 42
    model_: IsolationForest | None = field(default=None, repr=False)

    def fit(self, x: np.ndarray) -> IsolationForestDetector:
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.seed,
            n_jobs=-1,
        ).fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Anomaly score (higher = more anomalous); the negated sklearn decision function."""
        assert self.model_ is not None, "detector is not fitted"
        return -self.model_.decision_function(x)
