# Benchmark report — creditcard-random-split

Generated 2026-10-16T20:30:28+00:00 by `fraudshield train`. Model version `xgboost-20261016-203028`.

## Data and protocol

- Source file SHA-256 `76274b691b16a6c4…`, 284,807 rows, 492 frauds; 1,081 exact duplicates removed.
- Split: **stratified-random seed=42** (out-of-time).

| Split | Rows | Frauds | Fraud rate | Time window (h) |
|---|---:|---:|---:|---|
| train | 170,235 | 284 | 0.1668% | 0.0–48.0 |
| validation | 56,745 | 94 | 0.1657% | 0.0–48.0 |
| test | 56,746 | 95 | 0.1674% | 0.0–48.0 |

- selection_metric: validation PR-AUC
- selection_log: ['logreg: val PR-AUC gain -0.0785 vs xgboost (90% CI low -0.1322) -> does not qualify', 'stacked: val PR-AUC gain -0.0031 vs xgboost (90% CI low -0.0141) -> does not qualify', 'selected: xgboost']
- calibration: Platt scaling fit on validation
- thresholds: chosen on validation (cost-optimal review threshold; F1-optimal for comparisons)
- test_usage: scored once after all choices were frozen

### Model selection (validation only)

- logreg: val PR-AUC gain -0.0785 vs xgboost (90% CI low -0.1322) -> does not qualify
- stacked: val PR-AUC gain -0.0031 vs xgboost (90% CI low -0.0141) -> does not qualify
- selected: xgboost

## Test results (scored once)

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Recall @ FPR 0.1% | Recall @ FPR 1% | Precision @ top 0.1% |
|---|---|---|---:|---:|---:|
| XGBoost **(selected)** | 0.9745 [0.9491, 0.9929] | 0.8716 [0.8074, 0.9292] | 0.895 | 0.905 | 0.965 |
| Logistic regression | 0.9634 [0.9258, 0.9909] | 0.7464 [0.6482, 0.8337] | 0.821 | 0.895 | 0.877 |
| XGBoost (raw features) | 0.9749 [0.9491, 0.9926] | 0.8634 [0.7980, 0.9222] | 0.874 | 0.905 | 0.982 |
| Isolation Forest | 0.9507 [0.9213, 0.9748] | 0.1840 [0.1213, 0.2824] | 0.232 | 0.611 | 0.386 |
| Autoencoder | 0.9419 [0.9065, 0.9738] | 0.6604 [0.5553, 0.7640] | 0.811 | 0.821 | 0.807 |
| Stacked ensemble (OOF) | 0.9776 [0.9540, 0.9935] | 0.8722 [0.8086, 0.9306] | 0.884 | 0.905 | 0.965 |
| Stacked (legacy, in-sample) | 0.9735 [0.9462, 0.9924] | 0.8719 [0.8085, 0.9313] | 0.884 | 0.905 | 0.965 |

Paired bootstrap of test PR-AUC difference vs XGBoost (same resamples):

| Model | Δ PR-AUC | 95% CI | P(Δ ≤ 0) |
|---|---:|---|---:|
| Logistic regression | -0.1252 | [-0.2009, -0.0597] | 1.00 |
| XGBoost (raw features) | -0.0082 | [-0.0233, +0.0062] | 0.87 |
| Isolation Forest | -0.6876 | [-0.7582, -0.5912] | 1.00 |
| Autoencoder | -0.2112 | [-0.2979, -0.1261] | 1.00 |
| Stacked ensemble (OOF) | +0.0006 | [-0.0095, +0.0099] | 0.41 |
| Stacked (legacy, in-sample) | +0.0003 | [-0.0095, +0.0107] | 0.45 |

## Operating point of the selected model (XGBoost)

Review threshold **0.1548** (calibrated probability), chosen on validation as the cost-optimal threshold under the illustrative assumptions below.

| Metric | Test value |
|---|---:|
| Precision | 0.920 [0.854, 0.973] |
| Recall | 0.842 [0.772, 0.911] |
| F1 | 0.879 |
| False-positive rate | 0.00012 |
| Alerts | 87 of 56,746 (0.153%) |
| Confusion matrix | TP 80 · FP 7 · FN 15 · TN 56,644 |

## Calibration (test)

| | Brier | Log-loss | ECE |
|---|---:|---:|---:|
| Raw model output | 0.000323 | 0.00240 | 0.00020 |
| Platt-calibrated | 0.000342 | 0.00235 | 0.00027 |

## Cost experiment (illustrative assumptions)

Assumptions: review cost €5.00 per alert; a missed fraud loses 100% of its amount; reviewed fraud is caught. Illustrative assumptions; no real loss or review-cost data available.

| Policy on test window | Reviews | Frauds caught | Fraud loss (€) | Review cost (€) | Total (€) |
|---|---:|---:|---:|---:|---:|
| approve all | 0 | — | 12,973.06 | 0.00 | 12,973.06 |
| review all | 56,746 | — | 0.00 | 283,730.00 | 283,730.00 |
| threshold 0.5 | 78 | 76 | 2,596.10 | 390.00 | 2,986.10 |
| validation cost optimal | 87 | 80 | 2,464.68 | 435.00 | 2,899.68 |
| expected loss rule | 122 | 44 | 1,338.26 | 610.00 | 1,948.26 |
| test oracle threshold for reference only | 113 | 83 | 2,010.19 | 565.00 | 2,575.19 |

The *test oracle* row uses test labels to pick the threshold and is shown only as an upper bound.

## In-process latency (single transaction, this machine)

| Model | p50 (ms) | p95 (ms) | p99 (ms) | Batch throughput (rows/s) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.124 | 0.232 | 0.270 | 536,711 |
| XGBoost | 0.685 | 1.306 | 1.640 | 133,421 |
| Stacked ensemble (OOF) | 27.755 | 69.747 | 74.788 | 41,342 |

## Global feature attribution (mean |SHAP|, validation)

| Feature | mean abs SHAP (log-odds) |
|---|---:|
| V14 | 1.0383 |
| V4 | 0.9432 |
| V12 | 0.6437 |
| V15 | 0.3609 |
| V8 | 0.3421 |
| V10 | 0.3123 |
| V3 | 0.2736 |
| V26 | 0.2483 |
| V19 | 0.2260 |
| V24 | 0.2182 |
| V28 | 0.1991 |
| hour_sin | 0.1872 |

SHAP values describe how the model uses features; they are not causal evidence of fraud.

## Environment

fraudshield `1.0.0`, python `3.12.13`, platform `Windows-11-10.0.22000-SP0`, xgboost `3.4.1`, numpy `2.5.4`, pandas `3.0.6`, sklearn `1.9.1`, git_commit `c7e571f-dirty`

Runtime: 742.2 s.
