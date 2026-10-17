# Benchmark report — creditcard-chronological

Generated 2026-10-16T20:40:43+00:00 by `fraudshield train`. Model version `xgboost-20261016-204043`.

## Data and protocol

- Source file SHA-256 `76274b691b16a6c4…`, 284,807 rows, 492 frauds; 1,081 exact duplicates removed.
- Split: **chronological 60%/20%/20%** (out-of-time).

| Split | Rows | Frauds | Fraud rate | Time window (h) |
|---|---:|---:|---:|---|
| train | 170,235 | 342 | 0.2009% | 0.0–33.4 |
| validation | 56,745 | 57 | 0.1004% | 33.4–40.3 |
| test | 56,746 | 74 | 0.1304% | 40.3–48.0 |

- selection_metric: validation PR-AUC
- selection_log: ['logreg: val PR-AUC gain -0.0042 vs xgboost (90% CI low -0.0327) -> does not qualify', 'stacked: val PR-AUC gain +0.0015 vs xgboost (90% CI low -0.0103) -> does not qualify', 'selected: xgboost']
- calibration: Platt scaling fit on validation
- thresholds: chosen on validation (cost-optimal review threshold; F1-optimal for comparisons)
- test_usage: scored once after all choices were frozen

### Model selection (validation only)

- logreg: val PR-AUC gain -0.0042 vs xgboost (90% CI low -0.0327) -> does not qualify
- stacked: val PR-AUC gain +0.0015 vs xgboost (90% CI low -0.0103) -> does not qualify
- selected: xgboost

## Test results (scored once)

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Recall @ FPR 0.1% | Recall @ FPR 1% | Precision @ top 0.1% |
|---|---|---|---:|---:|---:|
| XGBoost **(selected)** | 0.9741 [0.9539, 0.9913] | 0.8014 [0.7131, 0.8812] | 0.811 | 0.851 | 0.912 |
| Logistic regression | 0.9804 [0.9655, 0.9930] | 0.7894 [0.7043, 0.8697] | 0.811 | 0.865 | 0.895 |
| XGBoost (raw features) | 0.9831 [0.9704, 0.9940] | 0.7906 [0.7049, 0.8720] | 0.797 | 0.865 | 0.895 |
| Isolation Forest | 0.9529 [0.9281, 0.9755] | 0.0578 [0.0412, 0.0828] | 0.000 | 0.581 | 0.000 |
| Autoencoder | 0.9351 [0.8966, 0.9678] | 0.2060 [0.1372, 0.3133] | 0.216 | 0.743 | 0.263 |
| Stacked ensemble (OOF) | 0.9714 [0.9498, 0.9901] | 0.8077 [0.7202, 0.8860] | 0.811 | 0.865 | 0.930 |
| Stacked (legacy, in-sample) | 0.9707 [0.9501, 0.9888] | 0.7989 [0.7114, 0.8815] | 0.811 | 0.838 | 0.930 |

Paired bootstrap of test PR-AUC difference vs XGBoost (same resamples):

| Model | Δ PR-AUC | 95% CI | P(Δ ≤ 0) |
|---|---:|---|---:|
| Logistic regression | -0.0120 | [-0.0445, +0.0199] | 0.74 |
| XGBoost (raw features) | -0.0108 | [-0.0345, +0.0089] | 0.82 |
| Isolation Forest | -0.7436 | [-0.8126, -0.6648] | 1.00 |
| Autoencoder | -0.5953 | [-0.6805, -0.4835] | 1.00 |
| Stacked ensemble (OOF) | +0.0064 | [-0.0025, +0.0176] | 0.09 |
| Stacked (legacy, in-sample) | -0.0024 | [-0.0118, +0.0063] | 0.75 |

## Operating point of the selected model (XGBoost)

Review threshold **0.5334** (calibrated probability), chosen on validation as the cost-optimal threshold under the illustrative assumptions below.

| Metric | Test value |
|---|---:|
| Precision | 0.945 [0.878, 1.000] |
| Recall | 0.703 [0.606, 0.803] |
| F1 | 0.806 |
| False-positive rate | 0.00005 |
| Alerts | 55 of 56,746 (0.097%) |
| Confusion matrix | TP 52 · FP 3 · FN 22 · TN 56,669 |

## Calibration (test)

| | Brier | Log-loss | ECE |
|---|---:|---:|---:|
| Raw model output | 0.000393 | 0.00254 | 0.00021 |
| Platt-calibrated | 0.000394 | 0.00255 | 0.00020 |

## Cost experiment (illustrative assumptions)

Assumptions: review cost €5.00 per alert; a missed fraud loses 100% of its amount; reviewed fraud is caught. Illustrative assumptions; no real loss or review-cost data available.

| Policy on test window | Reviews | Frauds caught | Fraud loss (€) | Review cost (€) | Total (€) |
|---|---:|---:|---:|---:|---:|
| approve all | 0 | — | 7,727.67 | 0.00 | 7,727.67 |
| review all | 56,746 | — | 0.00 | 283,730.00 | 283,730.00 |
| threshold 0.5 | 56 | 52 | 3,641.20 | 280.00 | 3,921.20 |
| validation cost optimal | 55 | 52 | 3,641.20 | 275.00 | 3,916.20 |
| expected loss rule | 81 | 24 | 2,441.70 | 405.00 | 2,846.70 |
| test oracle threshold for reference only | 96 | 60 | 2,389.79 | 480.00 | 2,869.79 |

The *test oracle* row uses test labels to pick the threshold and is shown only as an upper bound.

## In-process latency (single transaction, this machine)

| Model | p50 (ms) | p95 (ms) | p99 (ms) | Batch throughput (rows/s) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.120 | 0.247 | 0.312 | 704,667 |
| XGBoost | 0.548 | 0.835 | 1.201 | 349,824 |
| Stacked ensemble (OOF) | 19.090 | 25.485 | 29.000 | 53,271 |

## Global feature attribution (mean |SHAP|, validation)

| Feature | mean abs SHAP (log-odds) |
|---|---:|
| V4 | 0.9990 |
| V14 | 0.5718 |
| V12 | 0.5715 |
| V10 | 0.4871 |
| v_mean | 0.2119 |
| V8 | 0.1478 |
| Amount | 0.1343 |
| V13 | 0.1248 |
| V16 | 0.1205 |
| V11 | 0.1121 |
| hour_sin | 0.1046 |
| V20 | 0.1000 |

SHAP values describe how the model uses features; they are not causal evidence of fraud.

## Environment

fraudshield `1.0.0`, python `3.12.13`, platform `Windows-11-10.0.22000-SP0`, xgboost `3.4.1`, numpy `2.5.4`, pandas `3.0.6`, sklearn `1.9.1`, git_commit `c7e571f-dirty`

Runtime: 598.5 s.
