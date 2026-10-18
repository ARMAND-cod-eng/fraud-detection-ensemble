# FraudShield v2 — IEEE-CIS on authorisation-time fields (ieee-v2-raw-plus-velocity-20261017-171440)

590,540 transactions, chronological 60%/20%/20%; field contract of 24 raw fields; 15 velocity features.

| Variant | Val PR-AUC | Test PR-AUC [95% CI] | Δ vs raw [95% CI] | Test ROC-AUC | Recall @ FPR 1% |
|---|---:|---|---|---:|---:|
| raw contract | 0.277 | 0.208 [0.196, 0.222] | — | 0.806 | 16.2% |
| raw plus velocity **(selected)** | 0.287 | 0.217 [0.204, 0.231] | +0.009 [+0.001, +0.018] | 0.821 | 17.3% |

Reference: all 431 columns (incl. vendor C/D/V/id_ aggregates), test PR-AUC 0.508. Share of the raw-to-vendor gap recovered by velocity features: 3%.

Selection: velocity: val PR-AUC gain +0.0094 (90% CI low +0.0023) -> selected: raw_plus_velocity

Operating point (review threshold 0.167, chosen on validation): precision 26.3%, recall 31.1%, 4,809 alerts (4.07% of transactions).

Protocol amendment: 2026-10-17: review threshold capped at a 5% validation review rate after the first full run (uncapped cost-optimal threshold 0.021). Validation-only rule, but changed after a test run: test operating-point numbers are post-hoc.

Promotion gate: NOT promoted (artifact_integrity pass, smoke_test pass, test_pr_auc FAIL, test_roc_auc FAIL, p95_latency_ms pass).

Online latency per transaction (feature-store lookup + update + model): p50 0.79 ms, p95 1.35 ms (feature store alone p95 0.18 ms).

## Monthly backtest (label-maturity gap: one month)

| Scored month | Frauds | Static model PR-AUC | Retrained monthly PR-AUC |
|---:|---:|---:|---:|
| 3 | 3898 | 0.253 | 0.253 |
| 4 | 2850 | 0.221 | 0.240 |
| 5 | 2972 | 0.198 | 0.249 |
