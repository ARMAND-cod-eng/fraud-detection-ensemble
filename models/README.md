# Models

Trained models are **not committed**. They are produced by `fraudshield train` into
`artifacts/models/<version>/` (git-ignored) and tracked by `artifacts/registry.json`:

| File | Content |
|---|---|
| `booster.json` | XGBoost model (native JSON format) |
| `model.json` | Model kind, feature-builder profile, Platt calibration, signal quantiles |
| `autoencoder.npz` | Autoencoder weights for the NumPy forward pass (trained with PyTorch) |
| `iforest.joblib` | Isolation Forest (pickle; loaded only after SHA-256 verification) |
| `metadata.json` | Version, data hash, split, metrics, gate checks, library versions |
| `evaluation.json` | Full offline evaluation (curves, bootstrap CIs, costs, latency, SHAP importance) |
| `reference_profile.json` | Drift-detection reference distributions |
| `MANIFEST.json` | SHA-256 of every file above |

Note: an earlier version of this file listed `autoencoder.pth`, `isolation_forest.pkl` and
`xgboost_model.json`; those files were never committed. See `docs/PROJECT_AUDIT.md`.
