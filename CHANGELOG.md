# Changelog

All notable changes. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Coverage gate in CI (`--cov-fail-under=88`; measured 92%), `pip-audit` and `npm audit` steps,
  Dependabot (pip, npm, actions, docker) and a pre-commit configuration.
- Tests for the CLI, report generation, dataset-download guards and optional MLflow tracking.

### Fixed
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
