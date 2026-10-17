# System design: a real-time card-fraud decision service

*"Design a fraud detection system"* is a standard ML system-design interview question. This
document answers it with this repository: every component is mapped to code that exists, and the
parts that only make sense at larger scale are labelled **(at scale)**.

## 1. Requirements

| | Target | This repository (measured) |
|---|---|---|
| Decision latency | Inside the card-authorisation path: tens of ms end to end (Stripe reports < 100 ms) | Model 0.84 ms p95 in process; HTTP `POST /score` 3.1 ms p50 / 4.8 ms p95 (Linux container) |
| Throughput | Peak authorisation rate with headroom | ~240–290 req/s per single-worker instance; scale out horizontally **(at scale)** |
| Quality | Catch most fraud value with a fixed review budget and very few false declines | Out-of-time: 81% recall at 0.1% FPR; 94.5% precision at the chosen review threshold |
| Auditability | Every decision explainable and reproducible | Model + policy version, reasons and TreeSHAP stored per decision; immutable artifacts |
| Availability | Fail safe: a scoring outage must not block all payments | Timeouts + idempotent retries; `/ready` gates traffic; caller falls back to rules **(at scale)** |

## 2. Architecture

```
 card network ─► authorisation service ─► FraudShield /score ─► decision (approve / review / decline)
                                          │  validate · auth · rate-limit · timeout · idempotency
                                          │  features ─► model ─► calibration ─► policy (+ rules at scale)
                                          ├─► decision log (SQLite → Postgres at scale)
                                          ├─► analyst queue + explanations ─► analyst decisions = labels
                                          └─► metrics / drift ─► alerts
 chargebacks (30–90 days later) ─────────► label store ─► retraining ─► evaluation gate ─► registry
```

Code: `api/` (contract, auth, limits), `src/fraudshield/service.py` (orchestration),
`inference/scorer.py`, `decision_engine/`, `storage/store.py`, `monitoring/`, `training/`.

## 3. Data and labels

- **Labels are delayed and biased.** Chargebacks arrive weeks later; declined transactions never
  get a label; analyst decisions are fast but noisy. Train on *settled* periods only, and keep
  declined-transaction outcomes in mind when reading metrics (selection bias).
  - Implemented: replayed history keeps labels hidden until an analyst closes the case
    (simulated delayed labels).
  - Implemented: analyst decisions are stored as feedback labels.
- **Splits must be out-of-time:** older data trains, newer validates, newest tests. Here a random
  split overstated PR-AUC (0.872 vs 0.801). Implemented: `data/creditcard.py::chronological_split`.
- **Duplicates and leakage:** exact duplicates are removed before splitting, and fitted statistics
  come from train only (`docs/PROJECT_AUDIT.md`).

## 4. Features

| Family | Examples | Status |
|---|---|---|
| Transaction | amount, time of day, merchant/product, card network | Implemented (`features/engineering.py`) |
| Entity / velocity | transactions per card in 1 h / 24 h / 7 d, distinct cards per device, time since previous | Offline + windowed implemented and benchmarked on IEEE-CIS (`data/ieee_cis.py`); online store planned (v2) |
| Network / graph | shared devices/emails across accounts, connected components | Data-dependent: ULB has no identifiers; IEEE-CIS `DeviceInfo` is a device *model* |
| Consortium / vendor | issuer signals, device fingerprinting | Vendor columns (IEEE-CIS C/D/V) used only as a reference ceiling |

- **One feature definition for training and serving**, or train/serve skew follows; that skew
  was the original notebook's bug. Implemented for row-local features, with a test proving the
  single-row and batch paths are identical. For stateful velocity features this needs an online
  store with parity tests (planned, v2).
- **Prefer windows to cumulative counts.** Counts since the start of the data drift by
  construction; on IEEE-CIS they cost −0.010 PR-AUC, and trailing windows removed most of that.

## 5. Model

- **Start simple, earn complexity.** Logistic regression was within noise of XGBoost here, and the
  stacked ensemble added +0.006 PR-AUC at 30–40× latency, so a pre-registered rule kept XGBoost
  (`training/pipeline.py`).
- **Imbalance:** evaluate with PR-AUC, recall at fixed FPR and precision at a review budget, not
  accuracy. Re-weighting was a tuned hyper-parameter, not a default.
- **Calibration:** decisions use probabilities, so calibrate (Platt on validation) and check
  reliability on test.
- **At scale:** a DNN, or a wide-and-deep model with entity embeddings, trained on much more data
  (Stripe moved from XGBoost to DNNs for parallel training and transfer learning).

## 6. Decisioning

- **Policy separate from the model,** versioned, with every decision recording both versions
  (`decision_engine/policy.py`).
- Two thresholds (review / decline) chosen on validation under stated costs, a review-capacity
  limit, and an optional amount-aware expected-loss rule (`docs/DECISION_ENGINE.md`).
- **Rules + ML hybrid (at scale):** deterministic rules for known patterns and regulatory blocks,
  evaluated beside the model so analysts can see which fired. Today the policy layer is where
  rules would plug in.

## 7. Evaluation

| Stage | What | Here |
|---|---|---|
| Offline | Out-of-time PR-AUC, recall@FPR, precision@budget, bootstrap CIs, cost curves, calibration | `training/evaluate.py`, `benchmarks/results/` |
| Backtest | Rolling monthly windows with a label-maturity gap, static vs retrained | Planned (v2, IEEE-CIS covers 182 days) |
| Pre-deployment gate | Integrity, smoke test, minimum metrics, latency, no regression vs production | `training/promotion.py` |
| Shadow | New model scores live traffic without acting; compare score distributions and alerts | **(at scale)**; registry and decision log support it |
| Online | Holdout / A-B on approval rate, fraud loss, review volume | **(at scale)** |

## 8. Monitoring

- **Before labels arrive,** monitor proxies: input drift (PSI vs the training reference), score
  drift (fixed probability bins; deciles missed tail shifts), alert rate vs the validation
  baseline, data-quality warnings, latency, errors (`monitoring/`, `/metrics`, Grafana).
- **After labels mature,** monitor realised precision/recall per cohort, then trigger retraining
  through the gate. Never deploy automatically.

## 9. Retraining and governance

Retrain on a schedule (monthly is typical) on settled data. Compare the challenger with the
champion on the latest settled period, promote through the gate, and keep one-command rollback.
Models are immutable, checksummed artifacts with metadata (data hash, split, metrics, library
versions) and a model card (`docs/MODEL_CARD.md`).

## 10. Explainability and compliance

Per-decision TreeSHAP attributions in calibrated log-odds, with training percentiles, are shown to
analysts. They're labelled as model associations, not evidence. Every decision keeps an audit trail of
model, policy, reasons and analyst actions. **(At scale):** customer-facing reason codes, PII
minimisation and retention, and fairness monitoring where customer attributes exist.

## 11. Scaling path

| Concern | Now (one instance) | At scale |
|---|---|---|
| State (rate limits, replay, SSE fan-out) | In process | Redis |
| Decision log | SQLite (WAL) | Postgres / event log |
| Event transport | In-process replay with backpressure | Kafka (or Kinesis) with consumer groups |
| Entity features | Offline windows (IEEE-CIS) | Online store (Redis) fed by stream processing, parity-tested |
| Serving | 1 uvicorn worker, ~250–290 req/s | N stateless replicas behind a load balancer, autoscaled on p95 latency |
| Model delivery | File registry | S3 + signed artifacts, staged rollout with shadow traffic |

## 12. Trade-offs worth discussing in an interview

- **Simplest model that ties:** maintenance, latency and explainability beat +0.006 PR-AUC.
- **Out-of-time evaluation** costs optimism up front but avoids surprises in production.
- **Calibrated probabilities plus a policy layer** let business owners change thresholds safely.
- **Window features over cumulative counts:** stationarity matters more than richness.
- **One worker + SQLite** keep the demo reproducible. The interfaces are narrow so the scale-out
  path above is an implementation change, not a redesign.
