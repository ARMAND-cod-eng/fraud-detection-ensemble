# Project audit — `fraud-detection-ensemble` (state before the FraudShield upgrade)

Audit date: 2026-10-16. Audited commit: `c7e571f` (branch `main`).
Method: read every file in the repository, re-ran the original notebook's pipeline faithfully
(`benchmarks/reproduce_legacy.py`, 5 seeds) on the canonical dataset (SHA-256 `76274b69…`), and
measured each claim. All numbers below come from `benchmarks/results/legacy_reproduction.json`.

## 1. What the repository actually contained

| Path | Content | Status |
|---|---|---|
| `notebooks/01_fraud_detection_complete.ipynb` | Kaggle notebook: EDA, feature engineering, autoencoder, Isolation Forest, XGBoost, evaluation, "production pipeline" class, ROI table | The only source of code |
| `README.md` | Claims: ROC-AUC 0.9834, recall 85.71%, FPR 0.16%, ROI 1,779%, 86 ms, "production-ready" | Partly reproducible; see §4 |
| `models/README.md` | Describes `autoencoder.pth`, `isolation_forest.pkl`, `xgboost_model.json` | **No model files exist** |
| `notebooks/README.md` | Describes `02_business_impact.ipynb` | **Notebook does not exist** |
| `README.md` project tree | Lists `src/README.md` "coming soon" | **No `src/` directory** |
| `data/README.md` | Dataset description and Kaggle download command | Accurate |
| `requirements.txt` | Pins torch 2.0.1, numpy 1.24.3, xgboost 1.7.6, … | Not installable on Python ≥ 3.12; notebook code also fails on XGBoost ≥ 2.0 |
| `medical-insurance-cost-prediction.ipynb` (root) | Unrelated regression project | Should live in its own repository (left in place; not deleted) |

There was no API, no tests, no Docker, no CI, no saved artifacts, no configuration and no
deployment. "Production pipeline" meant a Python class inside the notebook.

## 2. Existing ML architecture

```
creditcard.csv ──► FeatureEngineer (fit on FULL dataset) ──► random stratified split 64/16/20
                                                                  │
                         RobustScaler (fit on train) ◄────────────┘
                                  │
      ┌───────────────────────────┼──────────────────────────────┐
      ▼                           ▼                              ▼
 Autoencoder (train normals,  Isolation Forest (train,      47 engineered features
 Sigmoid output, 50 epochs)   200 trees)                          │
      │ in-sample recon. error    │ decision_function             │
      └───────────────┬───────────┴───────────────────────────────┘
                      ▼
       XGBoost (300 trees, depth 6, scale_pos_weight, early stop on val)
                      ▼
          probability ≥ 0.5 → fraud (threshold fixed, never tuned on validation)
```

## 3. Technical problems found (with evidence)

| # | Problem | Evidence | Severity |
|---|---|---|---|
| P1 | **Autoencoder never learned.** RobustScaler output (e.g. `Amount_squared` up to ~6.6e8 before scaling) was reconstructed through a `Sigmoid` output bounded to (0, 1). | Training loss 1418.2 → 1418.1 over 50 epochs (notebook output and reproduction). Its "reconstruction error" is effectively a distance-from-zero of unbounded inputs. | High |
| P2 | **Preprocessing statistics fit on the full dataset** (incl. test): `FeatureEngineer.fit` ran on all 284,807 rows before splitting (z-score means/stds). | Cell 16 output: "fitted on 284807 samples". Label-free, so the effect is small, but it is leakage by construction. | Medium |
| P3 | **Duplicate leakage across the random split.** The file holds 1,081 exact duplicate rows (19 fraud). | 260–298 test rows per seed have an exact copy in the training set. | Medium |
| P4 | **Stacking on in-sample detector scores.** The autoencoder was trained on training normals and its errors on those same rows were fed to XGBoost: training normals look "more normal" than unseen ones, a label-dependent artefact. | By design (cell 24). Quantified by the new pipeline's `stacked_insample` vs. `stacked` comparison. | Medium |
| P5 | **Threshold/business choices made on the test set.** The ROI table and "optimal threshold" were computed on test predictions. | Cell 38 uses `y_test`. | Medium |
| P6 | **Inconsistent "optimal threshold".** Notebook calls 0.5 optimal although its own table shows higher net benefit at 0.3, and ROI rises monotonically with the threshold. | Cell 38 output (33,395 at 0.3 vs 32,765 at 0.5). | Low |
| P7 | **ROI depends on a $500 flat fraud amount.** The dataset has real amounts; mean fraud amount is €122. | Reproduction at threshold 0.5: flat-$500 assumption → "fraud prevented" 41,500, ROI 1,728%; actual amounts → 8,362 and **268%**. | High (claim) |
| P8 | **Train/serve skew in the "real-time" path.** `pipeline.predict` re-applied feature engineering to already-engineered rows (PCA stats then included `V*_zscore` columns), and single-row `np.std` (ddof 0) differs from batch `pandas.std` (ddof 1). | On 200 test rows: max \|Δp\| 0.032, 3 rows differ by > 0.01. | Medium |
| P9 | **Early stopping never triggered** (best iteration 299 of 300), so the model was capped, not tuned. | Reproduction `xgb_best_iteration = 299`. | Low |
| P10 | **No uncertainty reported.** 98 test frauds → recall moves ~1 point per fraud. | Across 5 split seeds: recall 77.6%–90.8%, ROC-AUC 0.977–0.990. | Medium |
| P11 | **Unverified narrative claims** ("45% improvement in F1 over single models", "feature engineering improved performance by ~15%", "industry benchmark" columns). No single-model baselines or ablations exist in the notebook. | Absent from code. | High (credibility) |
| P12 | **Latency is ~100 ms because of per-row pandas work and an unbatched Isolation Forest**, not model complexity. | Reproduction: p50 98.6 ms, p95 101.6 ms on this CPU (claimed 86 ms on Kaggle). | Low |
| P13 | Raw `Time` (seconds since capture start) used to derive "Hour" assuming `Time=0` is midnight — undocumented assumption; the 48 h window cannot validate seasonality. | Dataset documentation. | Low |
| P14 | Not runnable: notebook paths are Kaggle-specific; `early_stopping_rounds`/`use_label_encoder` in `fit()` fail on XGBoost ≥ 2; pins fail on Python ≥ 3.12. | Reproduction required API changes (documented in the script). | Medium |

Security: no secrets in the repository; no network surface existed. The email address in the README is public contact information.

## 4. Reproducibility assessment

Faithful port of the notebook (same features, split seed 42, hyper-parameters, 0.5 threshold); only
XGBoost API changes were required. Python 3.12.13, XGBoost 3.4.1, scikit-learn 1.9.1, PyTorch 2.14.1 (CPU).

| Metric (test, threshold 0.5) | README claim | Reproduced, seed 42 | 5 seeds mean ± sd [min, max] |
|---|---:|---:|---|
| ROC-AUC | 0.9834 | 0.9828 | 0.983 ± 0.005 [0.977, 0.990] |
| PR-AUC (not reported originally) | — | 0.817 | 0.820 ± 0.036 [0.775, 0.867] |
| Recall | 85.71% | 84.69% | 85.5% ± 5.0% [77.6%, 90.8%] |
| Precision | 47.46% | 46.37% | 48.2% ± 3.3% |
| F1 | 0.6109 | 0.5993 | 0.615 ± 0.031 |
| Single-transaction latency | 86 ms | 98.6 ms p50 | — |
| ROI @0.5 | 1,779% | 1,728% ($500 assumption) / **268% (actual amounts)** | — |

**Verdict.** The headline classification metrics *do* reproduce within seed noise under the
original protocol, so they were not fabricated. They are, however, measured with a random split
that leaks duplicates, a broken autoencoder, an untuned threshold and no confidence intervals,
and the business-impact numbers are driven by an assumption that the data contradicts. They
are not a reliable estimate of out-of-time performance (see `docs/ML_EVALUATION.md`).

## 5. What was worth preserving

* The **problem framing** (supervised booster + unsupervised anomaly signals) and the three
  model families — kept, fixed, and benchmarked honestly.
* The idea of **cost-aware thresholding** — kept, but moved to validation data with real amounts
  and explicit assumptions, and separated from the model as a policy layer.
* The notebook itself — kept unchanged as the historical artefact; superseded by the package.

## 6. Highest-impact improvements (ranked)

1. Leakage-free, out-of-time evaluation with baselines, ablations and confidence intervals
   (credibility is the whole point of a portfolio ML project).
2. A real, tested Python package + API with one feature code path for training and serving.
3. Calibrated probabilities + a separate, configurable decision policy with honest cost analysis.
4. A product surface: replay, live monitor, investigation workflow, model/monitoring views.
5. Operational basics: artifacts with checksums, registry with gated promotion/rollback,
   Prometheus metrics, drift detection, Docker, CI.
6. Documentation that only claims what was measured.

## 7. Implementation priorities, complexity and dependencies

| Phase | Work | Complexity | Depends on |
|---|---|---|---|
| 1 | Reproduction script, data validation, chronological split, model zoo, selection rule, calibration, bootstrap CIs, cost experiment | M–L | dataset |
| 2 | `src/fraudshield` package, artifact format, FastAPI service, tests, Docker | L | 1 |
| 3 | Decision engine, replay engine, SQLite case store, React dashboard, SHAP explanations | L | 2 |
| 4 | Prometheus metrics, PSI drift, latency/load benchmarks, registry gates, GitHub Actions | M | 2 |
| 5 | README, docs, model card, demo script, screenshots | M | 1–4 |
| 6 | Second dataset (IEEE-CIS) and graph features | M–L | Kaggle terms acceptance by the user |

Timebox used: the plan targets ~28 focused hours; Phase 6 is optional and must not block 1–5.
Progress against this plan is tracked in `docs/IMPLEMENTATION_PROGRESS.md`.
