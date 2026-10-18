# FraudShield AI — real-time fraud intelligence platform

[![CI](https://github.com/ARMAND-cod-eng/fraud-detection-ensemble/actions/workflows/ci.yml/badge.svg)](https://github.com/ARMAND-cod-eng/fraud-detection-ensemble/actions/workflows/ci.yml)
![Coverage 93%](https://img.shields.io/badge/coverage-93%25-brightgreen)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

**Problem.** Card fraud is rare (≈0.1–0.2% of payments), costly when missed and costly to
over-flag, and most public fraud models are evaluated in ways that would not survive production.

**Approach.** FraudShield scores each transaction behind a FastAPI service, turns calibrated
probabilities into auditable approve / review / decline decisions through a versioned policy, and
gives analysts a React investigation workspace with per-decision explanations. Every model choice
is made on validation data from an *earlier* period; the final test window is scored once.

**Result.** On the held-out future window: PR-AUC **0.801**, **81%** of fraud caught at a 0.1%
false-positive rate, **0.84 ms** per decision, and an honest finding: the original ensemble did
not beat a single XGBoost, so it was not shipped.

**v2.** A real-time velocity feature store on IEEE-CIS (590k transactions): one feature definition
served online and offline with **0 mismatches in 1.77M values**, linked activity and plain-language
reason codes. The velocity features earn a small but real gain, and the v2 model **fails its own
pre-registered quality gate**. [Why, and what it means](#fraudshield-v2-real-time-velocity-features-on-ieee-cis).

![FraudShield walkthrough: replay, live monitor, investigation, performance, policy, drift](docs/images/demo.gif)

## Results at a glance (verified, held-out future window)

| | |
|---|---|
| Test PR-AUC (XGBoost, out-of-time) | **0.801** [95% CI 0.713–0.881] |
| Test ROC-AUC | **0.974** [0.954–0.991] |
| Recall at 0.1% false-positive rate | **81.1%** |
| At the validation-chosen review threshold | **94.5%** precision · **70.3%** recall · 55 alerts in 56,746 transactions |
| Illustrative fraud loss in the test window | €7,728 → **€3,716** with the example policy |
| Inference, one transaction (in-process p95) | **0.84 ms** |
| HTTP `POST /score`, Linux container | p50 **3.1 ms** · p95 4.8 ms · ~288 req/s per worker |
| Second dataset (IEEE-CIS, 590k transactions, all 431 columns) | test PR-AUC **0.508** [0.492–0.522] · ROC-AUC 0.887 |
| v2: IEEE-CIS on authorisation-time fields + 15 online velocity features | test PR-AUC **0.217** [0.204–0.231]; velocity gain **+0.009** [+0.001, +0.018]; **failed** the pre-registered gate |
| v2 online/offline feature parity | **0 mismatches** in 1,771,620 values (feature engines) and 75,000 values (through the API) |
| v2 latency incl. feature-store read + update (in-process) | p50 **0.79 ms** · p95 1.35 ms |
| Engineering checks | **90 automated tests**, 93% coverage (CI gate 88%), ruff, mypy, pip-audit, npm audit, Dependabot |

Data: ULB credit-card dataset (284,807 transactions, 48 h, 492 frauds), chronological
60/20/20 split, duplicates removed, every choice made on validation, test scored once.
Full report: [docs/ML_EVALUATION.md](docs/ML_EVALUATION.md).

### The honest headline: the ensemble did not earn its place

This project started as a notebook claiming an autoencoder + Isolation Forest + XGBoost ensemble.
Re-running it faithfully reproduced its headline ROC-AUC (0.983 vs claimed 0.9834) but exposed a
non-learning autoencoder, duplicate leakage, test-set threshold choices and an ROI driven by an
assumed $500 per fraud (the data's median fraud is €9). Rebuilt with leakage controls:

| Model (test) | PR-AUC [95% CI] | Δ vs XGBoost [95% CI] | p95 latency |
|---|---|---|---:|
| **XGBoost (selected)** | 0.801 [0.713, 0.881] | — | 0.84 ms |
| Stacked ensemble, out-of-fold | 0.808 [0.720, 0.886] | +0.006 [−0.003, +0.018] | 25–40 ms |
| Logistic regression | 0.789 [0.704, 0.870] | −0.012 [−0.045, +0.020] | 0.25 ms |
| XGBoost without engineered features | 0.791 [0.705, 0.872] | −0.011 [−0.035, +0.009] | — |
| Autoencoder (unsupervised) | 0.206 [0.137, 0.313] | −0.595 | — |
| Isolation Forest (unsupervised) | 0.058 [0.041, 0.083] | −0.744 | — |

A pre-registered rule (beat XGBoost by ≥ 0.01 validation PR-AUC with a CI excluding zero) rejected
the ensemble: no measurable gain for 30–40× the single-transaction latency. The anomaly detectors are kept as analyst
context, not as scoring inputs. Same pipeline on a *random* split reports PR-AUC 0.872 — random
splits flatter this problem. See [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md).

### Second dataset: relationship features that drifted

The ULB data has no identifiers, so relationship ("graph") features were tested on
**IEEE-CIS** (590,540 transactions over 182 days, with card, address, e-mail and device-model
fields), as a separate benchmark:

| IEEE-CIS, out-of-time test | PR-AUC [95% CI] | Δ vs original columns [95% CI] |
|---|---|---|
| **Original columns (selected on validation)** | **0.508** [0.492, 0.522] | — |
| + entity counts since the start of the data | 0.498 [0.481, 0.512] | −0.010 [−0.015, −0.006] |
| + entity counts over trailing 24 h / 7 d windows | 0.505 [0.488, 0.518] | −0.004 [−0.008, +0.000] |

Cumulative counts only grow with time (test medians 3–4× training), a built-in train/serve
mismatch that hurt the model; trailing windows fixed the drift but added nothing, because the
vendor's own counting and time-delta columns already carry that signal. Details:
[benchmarks/results/ieee-cis/report.md](benchmarks/results/ieee-cis/report.md).

## FraudShield v2: real-time velocity features on IEEE-CIS

Most of IEEE-CIS's 431 columns are aggregates computed by the data provider from history a merchant
does not have. v2 asks the merchant-side question: **using only the 24 raw fields available when a
card is authorised, plus velocity features computed in real time, how good is the model, and do
the velocity features earn their place?**

* **One definition, two engines.** 15 velocity features (per card: counts over 1 h / 24 h / 7 d,
  amount spent, amount vs. 7-day mean, seconds since previous, distinct e-mail domains / devices /
  regions; per device model, region and e-mail domain: distinct cards) are declared once
  (`fraudshield.features.velocity`). A vectorised offline engine builds the training data; an online
  store serves them per transaction.
* **Parity proven, not assumed:** Hypothesis property tests, plus a full replay of the real test
  window (118,108 transactions) with **0 mismatches in 1,771,620 values**, and **0 in 75,000** when
  replayed through the running API.
* **Served as a second profile** (`FRAUDSHIELD_PROFILE=ieee`). New transactions read their features
  as of before the event, then update the store; retries never double-count. Alerts carry
  plain-language reasons (e.g. "14 earlier transactions on this card in the last 24 h (typical: 6)").
  Investigators get linked activity (same card, card + e-mail, card + region, device model).

| IEEE-CIS, out-of-time test (118,108 transactions) | PR-AUC [95% CI] | Δ vs raw [95% CI] | ROC-AUC |
|---|---|---|---:|
| Raw authorisation-time fields | 0.208 [0.196, 0.222] | — | 0.806 |
| **+ 15 velocity features (selected on validation)** | **0.217** [0.204, 0.231] | **+0.009** [+0.001, +0.018] | 0.821 |
| *Reference: all 431 columns incl. vendor aggregates* | *0.508* | | *0.887* |

| Monthly backtest (PR-AUC) | Month 3 | Month 4 | Month 5 |
|---|---:|---:|---:|
| Static model (trained once on months 0–1) | 0.253 | 0.221 | 0.198 |
| Retrained monthly (latest month excluded: labels not mature) | 0.253 | 0.240 | 0.249 |

What the numbers say:
* The velocity gain is statistically real but small: it closes **3%** of the gap to the vendor-feature
  model. A merchant's own stream does not contain what the vendor aggregates encode, and the data
  has no true card or account ID to key on.
* A model trained once **loses 22% of its PR-AUC in three months**; monthly retraining holds it.
* The model **failed the gate set before the run** (PR-AUC ≥ 0.30, ROC-AUC ≥ 0.85). The gate was not
  lowered. It is served locally only through a logged `--force` override, shown on the dashboard.
* One protocol amendment is disclosed: the cost-optimal review threshold flagged 40% of traffic, so
  it was capped at a 5% review rate (validation-only rule, changed after a test run, so post-hoc).

![FraudShield v2 performance page: variants vs. ceiling, monthly backtest, gate result](docs/images/performance_ieee.png)

Full write-up: [ML_EVALUATION.md §12](docs/ML_EVALUATION.md#12-fraudshield-v2-ieee-cis-on-authorisation-time-fields--online-velocity-features)
· [model card](docs/MODEL_CARD_IEEE_V2.md) · [architecture](docs/ARCHITECTURE.md#data-profiles-and-the-online-feature-store-v2).
IEEE-CIS data and models stay local (competition rules); the public demo uses the ULB profile.

## What the platform does

```mermaid
flowchart LR
  T[Transactions<br/>API or replay] --> A[FastAPI<br/>validation · auth · rate limit<br/>timeouts · idempotency]
  A --> M[Model<br/>features → XGBoost → Platt]
  M --> P[Decision policy<br/>approve · review · decline<br/>versioned]
  P --> S[(SQLite<br/>decisions · cases · audit)]
  P --> E[TreeSHAP explanation<br/>+ anomaly signals]
  S --> UI[React dashboard<br/>live SSE stream]
  A --> O[/metrics · drift PSI ·<br/>data quality/]
  R[(Registry<br/>gated promotion · rollback)] --> M
```

1. **Scores** transactions (`POST /api/v1/transactions/score`, `/batch`) with a calibrated probability.
2. **Decides** with a separate, versioned policy (review / decline thresholds, review capacity,
   cost assumptions) — every decision stores the model and policy version.
3. **Explains** alerts with exact TreeSHAP attributions plus training-distribution percentiles,
   explicitly labelled as model associations, not evidence of fraud.
4. **Streams** a controlled replay of real held-out (or clearly synthetic) transactions with
   pause/resume, speed control, backpressure and failure isolation.
5. **Investigates**: alert queue, decision reasons, signals, audit trail, analyst decisions and
   notes; replayed history reveals its true label only after the analyst decides.
6. **Monitors** latency, throughput, errors, input/score drift (PSI) and out-of-range inputs;
   exports Prometheus metrics (Grafana dashboard provisioned).
7. **Governs models**: immutable artifacts with SHA-256 manifests, an evaluation gate before
   promotion, one-command rollback, optional MLflow tracking.
8. **Computes velocity features online** (`ieee` profile): a point-in-time feature store with
   proven parity to training, linked activity for investigators and plain-language reason codes.

| Live monitor | Model performance |
|---|---|
| ![Live](docs/images/live.png) | ![Performance](docs/images/performance.png) |
| **Decision policy what-if** | **Monitoring under synthetic drift** |
| ![Policy](docs/images/policy.png) | ![Monitoring](docs/images/monitoring.png) |

## Quick start

**Requirements:** Python 3.12 (tested; 3.11+ intended), Node 20+ for the dashboard (built with Node 24), ~1.5 GB disk. CPU only.

```bash
git clone https://github.com/ARMAND-cod-eng/fraud-detection-ensemble.git && cd fraud-detection-ensemble
python -m venv .venv && . .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/dev.txt && pip install --no-deps -e .

fraudshield download-data        # Kaggle CLI if configured, else the public TF mirror; SHA-256 verified
fraudshield train --promote      # ~10 min on 8 cores (configs/training_fast.yaml: ~3 min)
cd dashboard && npm ci && npm run build && cd ..
uvicorn api.main:app --port 8000 # dashboard: http://localhost:8000 · API docs: /docs
```

**Docker** (after training once so `./artifacts` holds a promoted model):

```bash
docker compose up -d api                        # API + dashboard, non-root, read-only root fs
docker compose --profile monitoring up -d       # optional Prometheus :9090 + Grafana :3000
```

Then start a replay from the dashboard (Overview → *Transaction replay*). The 90-second walkthrough
is in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

**v2 (IEEE-CIS profile)**: accept the [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection),
put the Kaggle files in `data/raw/ieee-cis/`, then:

```bash
fraudshield train-ieee                                   # ~11 min; report in benchmarks/results/ieee-cis-v2/
python scripts/check_feature_parity.py                   # online vs offline velocity features, real test window
fraudshield models promote --artifacts artifacts/ieee --config configs/ieee_v2.yaml \
  --version <version> --force --reason "local demo"      # only needed because the model fails the gate
FRAUDSHIELD_PROFILE=ieee uvicorn api.main:app --port 8010
python scripts/check_served_parity.py --api http://127.0.0.1:8010   # parity through the running API
```

## API examples

```bash
curl -s localhost:8000/api/v1/transactions/score -H 'Content-Type: application/json' -d '{
  "transaction_id": "txn_000123", "amount": 149.62, "time_seconds": 43200,
  "features": {"V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
    "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
    "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02,
    "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02}}'
```

Response (real output from the trained model, numbers rounded):

```json
{
  "transaction_id": "txn_000123",
  "fraud_probability": 0.000257,
  "raw_model_score": 0.000364,
  "risk_level": "low",
  "recommended_action": "approve",
  "decision_reasons": ["fraud probability 0.03% below review threshold 53.34%"],
  "top_risk_factors": [],
  "expected_loss_eur": 0.0384,
  "review_capacity_exceeded": false,
  "model_version": "xgboost-20261016-204043",
  "policy_version": "example-balanced@f5c0d8dc",
  "scored_at": "2026-10-16T21:23:31.771+00:00",
  "inference_ms": 0.95,
  "explanation_available": true,
  "explanation_url": "/api/v1/transactions/txn_000123/explanation",
  "data_quality_warnings": [],
  "synthetic": false,
  "idempotent_replay": false,
  "processing_ms": 1.99
}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/transactions/score` · `/batch` | Score one / up to 500 transactions (idempotent on `transaction_id`) |
| `GET /api/v1/transactions` · `/{id}` · `/{id}/explanation` | List/filter, detail + audit trail, TreeSHAP explanation with plain-language reasons |
| `GET /api/v1/transactions/{id}/linked` | Linked activity: transactions sharing a card, card + e-mail, card + region or device model (`ieee` profile) |
| `POST /api/v1/transactions/{id}/review` | Analyst decision or note |
| `GET /api/v1/models/info` · `/evaluation` · `/versions` | Model metadata, offline report, registry history |
| `GET/PUT /api/v1/policy` · `POST /api/v1/policy/simulate` | Active policy, change it, what-if on held-out data |
| `POST /api/v1/replay/start` · `pause` · `resume` · `stop`, `PATCH /api/v1/replay` | Controlled replay |
| `GET /api/v1/overview` · `/monitoring` · `/stream` (SSE) | Dashboard aggregates and live events |
| `GET /api/v1/config` | Served profile, currency, replay scenarios, demo limits, data attribution |
| `GET /health` · `/ready` · `/metrics` | Liveness, readiness (model verified + store + policy), Prometheus |

Errors always use `{"error": {"code", "message", "request_id"}}`; no stack traces leave the service.

## Training, evaluation and model operations

```bash
fraudshield train [--config configs/training.yaml] [--split chronological|stratified_random] [--promote]
fraudshield train-ieee                       # v2: IEEE-CIS authorisation-time fields + velocity, backtest
fraudshield evaluate                         # reload production artifact, re-score test, compare (reproducibility)
fraudshield models list | show | promote --version V | rollback
python benchmarks/reproduce_legacy.py        # faithful re-run of the original notebook (5 seeds)
python benchmarks/policy_experiment.py       # threshold/cost experiment
python benchmarks/load_test.py               # HTTP load test against a running API
pytest -q --cov=src/fraudshield --cov=api   # 90 tests; trains tiny models on synthetic data, no dataset needed
python benchmarks/ieee_cis_benchmark.py      # second dataset (needs Kaggle IEEE-CIS files, ~50 min CPU)
```

## Repository layout

```
src/fraudshield/   data · features (incl. velocity store) · profiles · models · training · inference · decision_engine · monitoring · streaming · storage
api/               FastAPI app, routes, schemas, middleware (logging, request IDs, security headers, rate limiting)
dashboard/         React + TypeScript (Vite), hand-written SVG charts, no UI framework
tests/             unit · integration (API end to end) · performance (opt-in)
benchmarks/        legacy reproduction, policy experiment, load test, IEEE-CIS benchmark, results/
configs/           training.yaml, training_fast.yaml, ieee_v2.yaml, decision_policy.yaml
scripts/           feature-parity checks, screenshots / demo GIF, Hugging Face Space deploy
deploy/            Prometheus, Grafana provisioning, AWS ECS task-definition template
docs/              audit, architecture, evaluation, model card, decision engine, deployment, trade-offs, demo
notebooks/         original exploratory notebook (kept for reference, with an audit note)
```

## Operational considerations

* **One worker by design.** Replay state, SSE fan-out, rate-limit buckets, drift windows and (v2)
  the velocity feature store live in process. ~240–290 req/s per instance on a laptop CPU; scaling
  out means Redis (feature store, buckets, pub/sub) + Postgres.
* **Security:** strict input validation, optional API key for mutating endpoints, constant-time
  key check, per-client rate limiting, inference timeouts with idempotent retries, CSP/security
  headers, non-root read-only container, artifacts checksum-verified before any pickle is loaded,
  no feature values or analyst notes in logs. Not included: SSO/RBAC, TLS (see DEPLOYMENT).
* **Labels arrive late.** Live monitoring shows drift and alert-rate shifts, not "accuracy";
  performance on replayed history is labelled as simulated delayed labels.
* **AWS:** documented ECR → ECS Fargate (1 task) + S3 artifacts + CloudWatch design with a task
  definition template; **not provisioned** ([docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).

## Known limitations

* Two days of anonymised 2013 data; 57/74 frauds in validation/test → wide intervals.
* V1–V28 are PCA components without disclosed meaning; explanations cannot be phrased in business terms.
* Costs are assumptions; no real chargeback or review-cost data.
* No customer attributes → no fairness/subgroup analysis possible on this dataset.
* IEEE-CIS results come from a fixed XGBoost configuration (no tuning per variant) and one seed;
  its `DeviceInfo` field is a device model, not a device ID, and `card1`–`card5` is not a unique card,
  so no true device graph is possible and "per card" velocity pools several cards.
* The v2 model fails its quality gate; the `ieee` profile is a demonstration of the serving pattern.

## Planned improvements

* Redis-backed feature store and Postgres for multi-instance serving; shadow-mode (champion vs.
  challenger) serving, today only evaluated offline in the v2 backtest.
* Signed artifacts and S3-backed registry; SSO for the analyst UI.
* Seed-variance study for the stochastic models; conformal or Venn-Abers calibration comparison.

## Engineering trade-offs (short)

Chronological over random splits; PR-AUC over ROC-AUC; the simplest model that ties; native
TreeSHAP over the `shap` dependency; PyTorch for training only (NumPy inference, 725 MB image
without it); SQLite + in-process streaming over Kafka/Redis for a reviewer-runnable demo;
hand-written charts over chart libraries (88 kB gzipped UI). Details:
[docs/ENGINEERING_TRADEOFFS.md](docs/ENGINEERING_TRADEOFFS.md).

## Documentation

[Audit](docs/PROJECT_AUDIT.md) · [System design](docs/SYSTEM_DESIGN.md) · [Architecture](docs/ARCHITECTURE.md) ·
[ML evaluation](docs/ML_EVALUATION.md) · [Model card](docs/MODEL_CARD.md) · [Model card v2](docs/MODEL_CARD_IEEE_V2.md) ·
[Decision engine](docs/DECISION_ENGINE.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Trade-offs](docs/ENGINEERING_TRADEOFFS.md) · [Demo script](docs/DEMO_SCRIPT.md) ·
[Implementation log](docs/IMPLEMENTATION_PROGRESS.md) · [Resume bullets](docs/RESUME_BULLETS.md)

## Data and licence

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
Machine Learning Group — ULB and Worldline (Database Contents License). v2:
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta), used
locally under the competition rules. Neither dataset nor any IEEE-CIS model is redistributed in this
repository. Code: MIT.

**Author:** Armand Junior Dongmo Notue — [LinkedIn](https://www.linkedin.com/in/notue250/) ·
[Kaggle](https://www.kaggle.com/armandjunior) · notuearmand250@gmail.com
