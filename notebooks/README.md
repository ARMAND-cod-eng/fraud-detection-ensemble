# Notebooks

### `01_fraud_detection_complete.ipynb` (original, kept unchanged)

The original Kaggle notebook: EDA, feature engineering, autoencoder, Isolation Forest, XGBoost
stacking, evaluation and a business-impact table. It is preserved as the historical starting
point of the project. It does not run unmodified on current libraries (Kaggle paths; XGBoost ≥ 2
API changes).

* A faithful, runnable port that verifies its claims: `benchmarks/reproduce_legacy.py`.
* What the audit found (non-learning autoencoder, duplicate leakage, test-set threshold choices,
  ROI driven by a $500-per-fraud assumption): `docs/PROJECT_AUDIT.md`.
* The current, leakage-free pipeline: `src/fraudshield/training/pipeline.py`
  (`fraudshield train`), results in `docs/ML_EVALUATION.md`.

An earlier version of this file described a `02_business_impact.ipynb`; that notebook was never
committed. The cost analysis now lives in `benchmarks/policy_experiment.py` and
`docs/DECISION_ENGINE.md`.
