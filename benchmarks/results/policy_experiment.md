# Decision-policy experiment

Model `xgboost-20261016-204043`. Illustrative cost assumptions; thresholds searched on validation, costs reported on test.
Assumptions: review €5.0, false decline €15.0, loss rate 100%. Test window 7.7 h.

| Policy | Review ≥ | Decline ≥ | Validation total € | Test: reviewed | Test: declined | Test: frauds stopped | Test: legit declined | Test: fraud loss € | Test: review € | Test: friction € | Test total € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| approve everything | — | — | 12,231.09 | 0 | 0 | 0/74 | 0 | 7,727.67 | 0.00 | 0.00 | 7,727.67 |
| example policy (configs/decision_policy.yaml) | 0.5334 | 0.900 | 4,569.86 | 15 | 40 | 52/74 | 0 | 3,641.20 | 75.00 | 0.00 | 3,716.20 |
| review-only (never auto-decline) | 0.5334 | 1.000 | 4,694.86 | 55 | 0 | 52/74 | 0 | 3,641.20 | 275.00 | 0.00 | 3,916.20 |
| example + expected-loss review rule | 0.5334 | 0.900 | 4,909.86 | 73 | 40 | 54/74 | 0 | 2,407.67 | 365.00 | 0.00 | 2,772.67 |
| two thresholds tuned on validation | 0.4784 | 0.800 | 4,539.86 | 10 | 47 | 52/74 | 0 | 3,641.20 | 50.00 | 0.00 | 3,691.20 |

## Sensitivity of the cost-optimal thresholds to review cost (validation)

| Review cost € | Best review threshold | Best decline threshold | Reviews | Frauds stopped | Cost € |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.4784 | 0.8 | 10 | 43/57 | 4,499.86 |
| 2.5 | 0.4784 | 0.8 | 10 | 43/57 | 4,514.86 |
| 5.0 | 0.4784 | 0.8 | 10 | 43/57 | 4,539.86 |
| 10.0 | 0.4784 | 0.8 | 10 | 43/57 | 4,589.86 |
| 20.0 | 0.4784 | 0.8 | 10 | 43/57 | 4,689.86 |
| 50.0 | 0.4784 | 0.8 | 10 | 43/57 | 4,989.86 |

## Review capacity (test window, top-scored alerts reviewed first)

| Reviews / hour | Reviews | Frauds caught | Recall | Precision | Residual loss € |
|---:|---:|---:|---:|---:|---:|
| 5 | 38 | 38 | 51.4% | 100.0% | 4,436.17 |
| 10 | 77 | 58 | 78.4% | 75.3% | 3,394.06 |
| 25 | 191 | 60 | 81.1% | 31.4% | 2,389.79 |
| 50 | 383 | 63 | 85.1% | 16.4% | 2,353.03 |
| 100 | 766 | 64 | 86.5% | 8.4% | 1,718.73 |

Median fraud amount in the test window: €4.79; 53% of frauds are smaller than the €5 review cost, so a pure cost-minimising rule would ignore them. Small fraudulent amounts are often card-testing that precedes larger fraud, which this cost model does not capture.
