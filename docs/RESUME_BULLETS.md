# Resume bullets (evidence-based)

Every number below is reproducible from this repository (see `docs/ML_EVALUATION.md` and
`docs/IMPLEMENTATION_PROGRESS.md`). Use one set per application; keep the hedges.

## Machine Learning Engineer

* Rebuilt a notebook fraud model into a tested, containerised scoring service (FastAPI, XGBoost,
  SQLite, Docker): sub-millisecond in-process inference (p95 0.84 ms), 3.1 ms p50 end-to-end HTTP
  latency and ~290 req/s per worker, with idempotent scoring, timeouts, rate limiting and
  Prometheus metrics; 90 automated tests in CI (93% coverage).
* Built a reproducible training pipeline with out-of-time evaluation, bootstrap confidence
  intervals, Platt calibration, SHA-256-verified model artifacts, and an evaluation gate for
  promotion and rollback; a retrain reproduced every test metric exactly.
* Built a real-time velocity feature store (15 features declared once, a vectorised offline
  engine plus a per-entity online store) and proved train/serve parity with property tests and a
  full replay of 118k real transactions: 0 mismatches in 1.77M values (0 in 75k through the API);
  0.79 ms p50 per transaction including the feature read and update.
* Profiled and cut API overhead (pure-ASGI middleware, async dependencies) and replaced a biased
  asyncio load generator with a multi-process one, separating client cost from server latency.

## Applied AI Engineer

* Shipped an end-to-end fraud-intelligence product: real-time transaction replay with
  backpressure, a versioned decision-policy engine, TreeSHAP explanations, and a React/TypeScript
  investigator dashboard with live server-sent events and an audited analyst workflow.
* Separated model from business policy: calibrated probabilities drive configurable approve /
  review / decline rules; a policy with a validation-chosen threshold cut illustrative fraud loss on a held-out
  window from €7.7k to €3.7k with 55 alerts in 56.7k transactions.
* Generalised the platform into dataset profiles: a second model (IEEE-CIS) served through the
  same API and dashboard, with linked activity for investigators and alert reasons rendered as
  plain sentences ("14 earlier transactions on this card in the last 24 h (typical: 6)").
* Added production-style monitoring: PSI drift for inputs and scores (fixing a decile-binning
  blind spot found during testing), data-quality checks and simulated delayed labels.

## Fraud Data Scientist

* Audited a published fraud model: found duplicate leakage, a non-learning autoencoder and an ROI
  claim driven by an assumed $500 per fraud (data median €9); a faithful 5-seed reproduction
  confirmed the headline ROC-AUC (0.983) but not the business case.
* Benchmarked logistic regression, XGBoost, Isolation Forest, an autoencoder and stacked ensembles
  out-of-time with paired bootstrap tests; the ensemble added +0.006 PR-AUC (95% CI −0.003 to
  +0.018) at 30–40× the latency, so a pre-registered rule selected XGBoost (test PR-AUC 0.801, 81% recall
  at 0.1% FPR).
* Ran a second, separate benchmark on IEEE-CIS (590k transactions): showed that cumulative entity
  counts degrade out-of-time PR-AUC (−0.010, 95% CI −0.015 to −0.006) because they drift with time,
  fixed it with trailing-window features, and kept the simpler model when the fix added no gain.
* Measured what real-time velocity features buy on IEEE-CIS authorisation-time fields: +0.009
  PR-AUC (95% CI +0.001 to +0.018), about 3% of the gap to the vendor's 431-column model. Ran a
  monthly backtest with a label-maturity gap (a static model loses 22% of PR-AUC in three months;
  monthly retraining holds), and kept the pre-registered promotion gate when the model failed it.
* Quantified operating trade-offs: cost curves, review-capacity limits (10 → 100 reviews/hour adds
  6 of 74 frauds), and why cost-only rules miss small card-testing frauds (53% of test frauds
  under €5).
