# Benchmark report — creditcard-chronological-fast

Generated 2026-10-16T21:27:48+00:00 by `fraudshield train`. Model version `xgboost-20261016-212748`.

## Data and protocol

- Source file SHA-256 `76274b691b16a6c4…`, 284,807 rows, 492 frauds; 1,081 exact duplicates removed.
- Split: **chronological 60%/20%/20%** (out-of-time).

| Split | Rows | Frauds | Fraud rate | Time window (h) |
|---|---:|---:|---:|---|
| train | 170,235 | 342 | 0.2009% | 0.0–33.4 |
| validation | 56,745 | 57 | 0.1004% | 33.4–40.3 |
| test | 56,746 | 74 | 0.1304% | 40.3–48.0 |

- selection_metric: validation PR-AUC
- selection_log: ['logreg: val PR-AUC gain -0.0031 vs xgboost (90% CI low -0.0219) -> does not qualify', 'stacked: val PR-AUC gain -0.0025 vs xgboost (90% CI low -0.0095) -> does not qualify', 'selected: xgboost']
- calibration: Platt scaling fit on validation
- thresholds: chosen on validation (cost-optimal review threshold; F1-optimal for comparisons)
- test_usage: scored once after all choices were frozen

### Model selection (validation only)

- logreg: val PR-AUC gain -0.0031 vs xgboost (90% CI low -0.0219) -> does not qualify
- stacked: val PR-AUC gain -0.0025 vs xgboost (90% CI low -0.0095) -> does not qualify
- selected: xgboost

## Test results (scored once)

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Recall @ FPR 0.1% | Recall @ FPR 1% | Precision @ top 0.1% |
|---|---|---|---:|---:|---:|
| XGBoost **(selected)** | 0.9857 [0.9747, 0.9948] | 0.8009 [0.7017, 0.8799] | 0.797 | 0.865 | 0.912 |
| Logistic regression | 0.9804 [0.9649, 0.9934] | 0.7894 [0.6893, 0.8705] | 0.811 | 0.865 | 0.895 |
| XGBoost (raw features) | 0.9824 [0.9662, 0.9947] | 0.7995 [0.6994, 0.8785] | 0.797 | 0.865 | 0.912 |
| Isolation Forest | 0.9514 [0.9257, 0.9737] | 0.0490 [0.0333, 0.0717] | 0.014 | 0.554 | 0.018 |
| Autoencoder | 0.9590 [0.9278, 0.9838] | 0.3526 [0.2418, 0.4820] | 0.446 | 0.784 | 0.509 |
| Stacked ensemble (OOF) | 0.9804 [0.9672, 0.9922] | 0.7903 [0.6897, 0.8698] | 0.797 | 0.838 | 0.912 |
| Stacked (legacy, in-sample) | 0.9850 [0.9730, 0.9944] | 0.7918 [0.6934, 0.8720] | 0.784 | 0.865 | 0.912 |

Paired bootstrap of test PR-AUC difference vs XGBoost (same resamples):

| Model | Δ PR-AUC | 95% CI | P(Δ ≤ 0) |
|---|---:|---|---:|
| Logistic regression | -0.0115 | [-0.0372, +0.0106] | 0.80 |
| XGBoost (raw features) | -0.0014 | [-0.0121, +0.0083] | 0.54 |
| Isolation Forest | -0.7519 | [-0.8246, -0.6663] | 1.00 |
| Autoencoder | -0.4482 | [-0.5385, -0.3370] | 1.00 |
| Stacked ensemble (OOF) | -0.0106 | [-0.0203, -0.0018] | 0.99 |
| Stacked (legacy, in-sample) | -0.0091 | [-0.0193, +0.0002] | 0.97 |

## Operating point of the selected model (XGBoost)

Review threshold **0.3915** (calibrated probability), chosen on validation as the cost-optimal threshold under the illustrative assumptions below.

| Metric | Test value |
|---|---:|
| Precision | 0.902 [0.815, 0.967] |
| Recall | 0.743 [0.638, 0.841] |
| F1 | 0.815 |
| False-positive rate | 0.00011 |
| Alerts | 61 of 56,746 (0.107%) |
| Confusion matrix | TP 55 · FP 6 · FN 19 · TN 56,666 |

## Calibration (test)

| | Brier | Log-loss | ECE |
|---|---:|---:|---:|
| Raw model output | 0.000405 | 0.00250 | 0.00029 |
| Platt-calibrated | 0.000402 | 0.00245 | 0.00019 |

## Cost experiment (illustrative assumptions)

Assumptions: review cost €5.00 per alert; a missed fraud loses 100% of its amount; reviewed fraud is caught. Illustrative assumptions; no real loss or review-cost data available.

| Policy on test window | Reviews | Frauds caught | Fraud loss (€) | Review cost (€) | Total (€) |
|---|---:|---:|---:|---:|---:|
| approve all | 0 | — | 7,727.67 | 0.00 | 7,727.67 |
| review all | 56,746 | — | 0.00 | 283,730.00 | 283,730.00 |
| threshold 0.5 | 58 | 52 | 2,684.83 | 290.00 | 2,974.83 |
| validation cost optimal | 61 | 55 | 2,678.17 | 305.00 | 2,983.17 |
| expected loss rule | 98 | 24 | 2,441.70 | 490.00 | 2,931.70 |
| test oracle threshold for reference only | 78 | 59 | 2,397.79 | 390.00 | 2,787.79 |

The *test oracle* row uses test labels to pick the threshold and is shown only as an upper bound.

## In-process latency (single transaction, this machine)

| Model | p50 (ms) | p95 (ms) | p99 (ms) | Batch throughput (rows/s) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.147 | 0.271 | 0.418 | 770,909 |
| XGBoost | 0.566 | 0.945 | 1.598 | 200,073 |
| Stacked ensemble (OOF) | 10.153 | 13.305 | 15.475 | 67,774 |

## Global feature attribution (mean |SHAP|, validation)

| Feature | mean abs SHAP (log-odds) |
|---|---:|
| V4 | 1.3936 |
| V14 | 1.0357 |
| V12 | 0.8756 |
| V10 | 0.5246 |
| V8 | 0.4091 |
| Amount | 0.3184 |
| V11 | 0.2750 |
| hour_sin | 0.2335 |
| V28 | 0.2311 |
| V22 | 0.2170 |
| V16 | 0.1920 |
| V19 | 0.1769 |

SHAP values describe how the model uses features; they are not causal evidence of fraud.

## Environment

fraudshield `1.0.0`, python `3.12.13`, platform `Windows-11-10.0.22000-SP0`, xgboost `3.4.1`, numpy `2.5.4`, pandas `3.0.6`, sklearn `1.9.1`, git_commit `c7e571f-dirty`

Runtime: 192.5 s.
