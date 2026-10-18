# Implementation progress

v1 on branch `fraudshield-upgrade` (from `main` @ `c7e571f`, 2026-10-16), merged; quick wins and
FraudShield v2 on 2026-10-17.
Hardware: Windows 11 laptop, 8 CPU cores, 8 GB RAM, no GPU. Python 3.12.13, Node 24, Docker 29.6.

## Status by phase

| Phase | Status | Evidence |
|---|---|---|
| 1. Scientific credibility | **Done** | `docs/PROJECT_AUDIT.md`, `benchmarks/results/legacy_reproduction.json`, `benchmarks/results/creditcard-chronological/`, `docs/ML_EVALUATION.md` |
| 2. Engineering foundation | **Done** | `src/fraudshield/`, `api/`, 58 passing tests, ruff + mypy clean, Docker image built and healthy |
| 3. Product experience | **Done** | Replay engine, decision engine, investigation workflow, React dashboard (screenshots in `docs/images/`) |
| 4. Operational credibility | **Done (local)** | Load tests, Prometheus metrics, PSI drift, registry gate/rollback, MLflow (optional), CI workflow file |
| 5. Portfolio delivery | **Done** | README, 9 docs, demo script, resume bullets |
| 6. Advanced research | **Done** | IEEE-CIS benchmark run on real data (2026-10-17): cumulative entity counts hurt (drift), windowed counts fixed the drift but added nothing; see `benchmarks/results/ieee-cis/report.md` |
| 7. FraudShield v2 (flagship) | **Done (local)** | Velocity feature store with proven parity, IEEE-CIS authorisation-time model, monthly backtest, `ieee` serving profile with linked activity and reason codes; `benchmarks/results/ieee-cis-v2/`, `docs/ML_EVALUATION.md` §12 |

## Log

1. **Audit.** Read every file; dumped the notebook; found no `src/`, no models, a missing
   `02_business_impact.ipynb`, an unrelated root notebook. Wrote `docs/PROJECT_AUDIT.md`.
2. **Data.** Downloaded the public copy of the ULB dataset, verified 284,807 rows / 492 frauds,
   pinned SHA-256 `76274b69…`, found 1,081 exact duplicates and a non-stationary fraud rate.
3. **Legacy reproduction** (`benchmarks/reproduce_legacy.py`, 5 seeds, ~8.5 min/seed on CPU):
   ROC-AUC 0.9828 (claimed 0.9834), recall 84.7% (claimed 85.7%); autoencoder loss stuck at 1418;
   260–298 duplicated rows leak into test; train/serve skew up to 0.032; ROI 1,728% only under a
   $500/fraud assumption (268% with real amounts); single-row latency 98.6 ms.
4. **Package** `src/fraudshield`: data validation/splits, shared feature builder, models
   (XGBoost, logistic, Isolation Forest, autoencoder with NumPy inference), Platt calibration,
   TreeSHAP, artifact format with SHA-256 manifest, file registry, evaluation library, training
   pipeline with pre-registered selection, report + figures, CLI.
5. **Training run** (598 s): XGBoost selected; ensemble rejected by the rule; test PR-AUC 0.801.
   A second identical run (after fixing PR-curve plotting and the drift reference) reproduced every
   metric exactly; `fraudshield evaluate` re-scored the saved artifact and matched the stored metrics exactly.
   Random-split comparison run: PR-AUC 0.872 (optimistic protocol).
6. **API** (FastAPI): scoring, batch, listing, detail, explanation, analyst review, model info,
   evaluation, registry, policy get/put/simulate, overview, monitoring, replay control, SSE stream,
   health/ready/metrics; error envelope; API key; rate limiting; inference timeout; idempotency.
7. **Dashboard** (React 19 + TypeScript 7 + Vite 8, no UI/chart libraries): overview, live monitor,
   investigations, model performance, decision policy, monitoring; light/dark; responsive
   (no horizontal scroll at 375 px).
8. **Performance work.** Profiled HTTP latency: replaced `BaseHTTPMiddleware` with pure ASGI
   middleware and made non-blocking dependencies `async` (FastAPI ran each sync dependency in the
   thread pool): sync-client p50 5.7 → 3.9 ms. Rewrote the load generator to use one process per
   client after measuring ~5 ms of client-side overhead in the asyncio client.
9. **Monitoring fixes found by using the product:** score PSI used validation deciles and missed a
   tail shift (switched to fixed log-spaced probability bins); hour-of-day features always "drifted"
   in partial-day windows (now shown but excluded from the overall status).
10. **Policy experiment** (`benchmarks/policy_experiment.py`) and documentation of the
    validation-vs-test disagreement on the expected-loss rule.
11. **Docker:** multi-stage image (725 MB, no PyTorch), Compose with read-only root fs; container
    reached `healthy` and served the model; in-container load test.
12. **MLflow:** optional; verified a tracked run (124 metrics, 11 params) on synthetic data. Found
    that `mlflow-skinny` cannot use SQLite and MLflow 3 gates the file store — fixed by an explicit
    file-store opt-in and by making tracking failures non-fatal.
13. **IEEE-CIS (2026-10-17):** after the user accepted the Kaggle rules and downloaded the data,
    ran the out-of-time benchmark (590,540 rows; ~35 min per two variants). Cumulative entity counts
    reduced test PR-AUC (0.508 → 0.498, CI excludes 0); diagnosed as drift (test medians 3–4× train).
    Added trailing-window counts (unit-tested against hand-computed values): 0.505, CI touches 0.
    Validation selects the original columns. Baseline reproduced exactly across runs.
14. **Quick wins (2026-10-17):** public demo mode + Hugging Face Space Dockerfile (built and run
    locally; publishing is the user's step), README GIF, `docs/SYSTEM_DESIGN.md`, CI coverage gate,
    pip-audit, npm audit, Dependabot, pre-commit, CHANGELOG, tag `v1.0.0`.
15. **v2 feature store (2026-10-17):** `features/velocity.py` with an offline engine and an online
    store. Hypothesis found two float edge cases (a near-zero window total crossing the ratio
    threshold; cross-entity cumulative-sum cancellation), both fixed. Real-data parity:
    0 mismatches in 1,771,620 values.
16. **v2 experiments (2026-10-17):** `fraudshield train-ieee` (11 min). Velocity adopted on validation
    (+0.0094, 90% CI low +0.0023); test 0.208 → 0.217, gain CI [+0.001, +0.018]; 3% of the gap to
    the 431-column ceiling. Backtest: static 0.253 → 0.198, retrained 0.24–0.25. The first run
    crashed at promotion (gate failed) before writing its report; the pipeline now records a failed
    gate as a result. The second run reproduced every metric exactly. The uncapped cost-optimal
    threshold flagged 40% of traffic, so it was capped at 5% (disclosed amendment, post-hoc on test).
17. **v2 serving (2026-10-17):** dataset profiles, IEEE request contract, online features in
    `FraudService`, `entity_links` + linked-activity endpoint, reason sentences, NaN-aware drift,
    profile-aware dashboard. The model was force-promoted for local serving with a logged reason.
    A 5,000-transaction replay through the running API matched the offline features exactly
    (75,000 values); reasons were reviewed on real alerts and percentiles removed where meaningless
    (category codes, masked IDs).

## Validation performed

| Check | Result |
|---|---|
| `pytest -q` | **90 passed**, 2 skipped (performance, opt-in) in ~100 s; coverage 93% |
| `FRAUDSHIELD_RUN_PERF=1 pytest tests/performance` | 2 passed (in-process p95 < 10 ms, HTTP p95 < 50 ms) |
| `ruff check .` / `ruff format --check .` | clean |
| `mypy` (src + api, 53 files) | no issues |
| `scripts/check_feature_parity.py` (real IEEE-CIS test window) | 0 mismatches / 1,771,620 values |
| `scripts/check_served_parity.py` (API, `ieee` profile) | 0 mismatches / 75,000 values |
| `npm run build` (tsc --noEmit + vite) | success, 288 kB JS (88 kB gzip) |
| `docker build --target runtime` + `docker compose up -d api` | built; container healthy; `/ready` OK |
| `fraudshield evaluate` | reproduced stored test metrics exactly |
| Fast config on real data | 192.5 s, test PR-AUC 0.801 |

## Not verified / remaining

* **CI on GitHub** runs on every push and pull request (tests with the coverage gate, lint, types,
  audits, dashboard build, Docker build); it ran green for v1. Check the v2 commits' run on GitHub.
* **Hugging Face Space:** the image was built and run locally; publishing it needs the user's own
  Hugging Face account and token (`scripts/deploy_hf_space.ps1`).
* **`trainer` Docker target and the `monitoring` Compose profile** (Prometheus/Grafana) were not
  built or started (disk/RAM limits on this machine); configs are provided, untested.
* **AWS:** design + task-definition template only; nothing provisioned.
* **IEEE-CIS:** one seed, fixed hyper-parameters. The v2 model is served only locally, through a
  logged override of its failed gate; the feature store is single-instance (Redis path documented).
* Seed variance for the autoencoder/stacked models not quantified; XGBoost `hist` is deterministic.
* The MLflow run for the production model was not recorded (MLflow was installed after it);
  re-running `fraudshield train` with `pip install -e .[tracking]` records it.
* `medical-insurance-cost-prediction.ipynb` (unrelated project) was removed from the repository on
  2026-10-17 (a local copy was kept outside the repo; it remains in git history).

## Next actions (suggested order)

1. Publish the Hugging Face Space and add the live-demo link to the README.
2. Record a 90-second screen capture following `docs/DEMO_SCRIPT.md`.
3. Optional: Redis-backed feature store for multi-instance serving; shadow (champion/challenger)
   serving using the backtest's retraining cadence.
4. Optional: deploy to AWS following `docs/DEPLOYMENT.md` (costs money; your decision).
