# Implementation progress

Branch `fraudshield-upgrade` (from `main` @ `c7e571f`). Work session: 2026-10-16.
Hardware: Windows 11 laptop, 8 CPU cores, 8 GB RAM, no GPU. Python 3.12.13, Node 24, Docker 29.6.

## Status by phase

| Phase | Status | Evidence |
|---|---|---|
| 1. Scientific credibility | **Done** | `docs/PROJECT_AUDIT.md`, `benchmarks/results/legacy_reproduction.json`, `benchmarks/results/creditcard-chronological/`, `docs/ML_EVALUATION.md` |
| 2. Engineering foundation | **Done** | `src/fraudshield/`, `api/`, 55 passing tests, ruff + mypy clean, Docker image built and healthy |
| 3. Product experience | **Done** | Replay engine, decision engine, investigation workflow, React dashboard (screenshots in `docs/images/`) |
| 4. Operational credibility | **Done (local)** | Load tests, Prometheus metrics, PSI drift, registry gate/rollback, MLflow (optional), CI workflow file |
| 5. Portfolio delivery | **Done** | README, 9 docs, demo script, resume bullets |
| 6. Advanced research | **Partially done** | IEEE-CIS loader + past-only entity/graph features + benchmark script, unit-tested on synthetic data; **not run on real data** (Kaggle terms) |

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

## Validation performed

| Check | Result |
|---|---|
| `pytest -q` | **55 passed**, 2 skipped (performance, opt-in) in ~25 s |
| `FRAUDSHIELD_RUN_PERF=1 pytest tests/performance` | 2 passed (in-process p95 < 10 ms, HTTP p95 < 50 ms) |
| `ruff check .` / `ruff format --check .` | clean |
| `mypy` (src + api, 46 files) | no issues |
| `npm run build` (tsc --noEmit + vite) | success, 288 kB JS (88 kB gzip) |
| `docker build --target runtime` + `docker compose up -d api` | built; container healthy; `/ready` OK |
| `fraudshield evaluate` | reproduced stored test metrics exactly |
| Fast config on real data | 192.5 s, test PR-AUC 0.801 |

## Not verified / remaining

* **CI on GitHub:** `.github/workflows/ci.yml` is written but has not run (nothing was pushed).
  Its steps mirror the commands above, which pass locally.
* **`trainer` Docker target and the `monitoring` Compose profile** (Prometheus/Grafana) were not
  built or started (disk/RAM limits on this machine); configs are provided, untested.
* **AWS:** design + task-definition template only; nothing provisioned.
* **IEEE-CIS:** not downloaded or run; requires your Kaggle account to accept competition rules.
* Seed variance for the autoencoder/stacked models not quantified; XGBoost `hist` is deterministic.
* The MLflow run for the production model was not recorded (MLflow was installed after it);
  re-running `fraudshield train` with `pip install -e .[tracking]` records it.
* `medical-insurance-cost-prediction.ipynb` (unrelated project) is still in the repository root;
  recommended: move it to its own repository.

## Next actions (suggested order)

1. Review and commit the branch; push and let CI run.
2. Accept the IEEE-CIS rules on Kaggle, run `python benchmarks/ieee_cis_benchmark.py`, add results.
3. Record a 90-second screen capture following `docs/DEMO_SCRIPT.md`.
4. Optional: deploy to AWS following `docs/DEPLOYMENT.md` (costs money; your decision).
