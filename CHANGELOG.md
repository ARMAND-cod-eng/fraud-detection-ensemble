# Changelog

All notable changes. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **FraudShield v2 (IEEE-CIS).** An authorisation-time field contract (24 raw fields, no vendor
  aggregates) and 15 velocity features declared once (`features/velocity.py`), with a vectorised
  offline engine and an `OnlineFeatureStore`.
  - Parity: Hypothesis property tests; `scripts/check_feature_parity.py` (0 mismatches in 1,771,620
    values on the real test window); `scripts/check_served_parity.py` (0 in 75,000 through the API).
  - `fraudshield train-ieee`: raw vs raw + velocity with a pre-registered selection rule, Platt
    calibration, monthly backtest with a label-maturity gap, gated artifact, replay and warm-up files.
- **Serving profiles** (`FRAUDSHIELD_PROFILE=creditcard|ieee`). The service, replay engine, API
  schemas and dashboard are dataset-agnostic, and `GET /api/v1/config` reports the served profile.
  The `ieee` profile adds:
  - online velocity features (read before the event, then update; retries never double-count);
  - late-event and unseen-category warnings;
  - linked activity (`GET /api/v1/transactions/{id}/linked`, `entity_links` table);
  - plain-language reasons (`top_risk_factors`).
- Dashboard: IEEE performance page (variants vs. ceiling, backtest, gate result), linked-activity
  panel, risk reasons in the alert queue and live feed, profile-aware currency and replay scenarios.
- PSI drift has a separate missing-value bin (older references are still read as before).
- `docs/MODEL_CARD_IEEE_V2.md`; `docs/ML_EVALUATION.md` §12; architecture section on profiles and the
  feature store.
- Coverage gate in CI (`--cov-fail-under=88`; measured 92%), `pip-audit` and `npm audit` steps,
  and a pre-commit configuration.
- Tests for the CLI, report generation, dataset-download guards and optional MLflow tracking.

### Results (verified, IEEE-CIS v2, out-of-time test)
- Raw contract PR-AUC 0.208 [0.196, 0.222]. Raw + velocity: 0.217 [0.204, 0.231], paired gain
  +0.009 [+0.001, +0.018], which recovers 3% of the gap to the all-431-column model (0.508).
- Monthly backtest: static model 0.253 → 0.198 PR-AUC over three months; monthly retraining holds
  0.24–0.25.
- The model **failed** the pre-registered promotion gate (PR-AUC ≥ 0.30, ROC-AUC ≥ 0.85). It is not
  promoted by the pipeline; local serving uses a logged `--force` override.
- Disclosed amendment: the review threshold is capped at a 5% validation review rate (the uncapped
  cost-optimal threshold flagged 40% of traffic), so test operating-point numbers are post-hoc.

### Fixed
- `run_ieee_v2` records a failed promotion gate in the report instead of aborting before the report
  and replay files are written.
- The promotion gate now applies the limits of the gate config in force at promotion time
  (previously the limits recorded at training time), so tightening the gate takes effect.
- `fraudshield models rollback` with no previous version returns an error instead of a traceback.

## [1.0.0] — 2026-10-17

First release of FraudShield AI, rebuilt from the original exploratory notebook.

### Added
- Audit and faithful 5-seed reproduction of the original notebook (`docs/PROJECT_AUDIT.md`).
- `fraudshield` package: validated data loading, out-of-time splits, shared feature builder,
  logistic regression / XGBoost / Isolation Forest / autoencoder / out-of-fold stacked ensemble,
  pre-registered model selection, Platt calibration, TreeSHAP, bootstrap CIs, cost and capacity
  analysis, SHA-256-verified artifacts, gated promotion and rollback.
- FastAPI scoring service with validation, optional API key, rate limiting, inference timeouts,
  idempotent retries, SQLite audit trail, Prometheus metrics and PSI drift monitoring.
- Transaction replay engine, versioned decision policy, React + TypeScript investigator dashboard.
- Docker image (no PyTorch), Compose, GitHub Actions CI, AWS ECS task-definition template.
- Second, separate benchmark on IEEE-CIS with past-only entity features (cumulative counts drift
  and hurt; windowed counts are neutral).

### Results (verified)
- ULB, out-of-time test: XGBoost PR-AUC 0.801 [0.713, 0.881], ROC-AUC 0.974; the ensemble's
  +0.006 PR-AUC is within noise at 30–40× the latency.
- IEEE-CIS, out-of-time test: PR-AUC 0.508 [0.492, 0.522].

[Unreleased]: https://github.com/ARMAND-cod-eng/fraud-detection-ensemble/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ARMAND-cod-eng/fraud-detection-ensemble/releases/tag/v1.0.0
