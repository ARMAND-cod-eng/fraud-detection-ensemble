# IEEE-CIS benchmark (second dataset)

Separate from the ULB credit-card benchmark; never combine their metrics.

* Data: Kaggle *IEEE-CIS Fraud Detection* (Vesta), `train_transaction.csv` + `train_identity.csv`,
  590,540 transactions over 182 days, 3.50% fraud (test window 3.44%). The Kaggle test set has no
  labels and is not used.
* Protocol: chronological 60/20/20 split on `TransactionDT` (354,324 / 118,108 / 118,108 rows);
  categorical codes learned on train only; XGBoost (depth 8, lr 0.05, colsample 0.5, early stopping
  on validation PR-AUC), identical settings for every variant; 500 paired bootstrap resamples on test.
* Selection rule (fixed before running): best **validation** PR-AUC; test reported for all.
* Runs: `metrics_run1_cumulative.json` (baseline + cumulative) and `metrics.json` (baseline +
  windowed). The baseline reproduced exactly across runs (test PR-AUC 0.50818 both times).

## Results

| Variant | Inputs | Val PR-AUC | Test PR-AUC [95% CI] | Δ vs original [95% CI] | Test ROC-AUC | Recall @ FPR 0.1% / 1% | Precision @ top 0.1% (118) |
|---|---:|---:|---|---|---:|---|---:|
| **Original columns (selected)** | 431 | **0.627** | **0.508** [0.492, 0.522] | — | 0.887 | 23.8% / 42.7% | 99.2% |
| + cumulative entity counts | 443 | 0.615 | 0.498 [0.481, 0.512] | −0.010 [−0.015, −0.006] | 0.892 | 21.8% / 42.2% | 94.9% |
| + windowed entity counts | 448 | 0.623 | 0.505 [0.488, 0.518] | −0.004 [−0.008, +0.000] | 0.890 | 22.9% / 42.3% | 100.0% |

Entity features (all past-only, never labels): per card, card + address, card + e-mail domain,
`DeviceInfo` and e-mail domain — transaction counts, seconds since the previous transaction, and
distinct cards per device / e-mail domain.

## Findings

1. **Cumulative counts drift by construction.** Counted from the start of the 182-day capture,
   they only grow: the test-period median is 3–4× the training median (e.g. prior transactions per
   card 174 → 756), and 25–39% of test rows exceed the training 90th percentile. The model learned
   thresholds on early, small counts; the features reduced PR-AUC significantly.
2. **Trailing windows fix the drift** (24 h / 7 day counts stay comparable across periods) and
   remove most of the damage, but still add nothing measurable: the paired CI touches zero and
   validation also prefers the original columns.
3. **Why no gain:** the vendor already ships counting, time-delta and entity-relation features
   (columns `C1–C14`, `D1–D15`, `V1–V339` per the competition's data description), so these features
   are largely redundant; they receive 1.5% of the model's split gain.
4. **`DeviceInfo` is a device model string** ("Windows", "iOS Device", "SM-G960U"), not a device
   identifier, so "shared device" means "shared device model". A meaningful device or account graph
   is not available in this data, which is why no graph neural network was attempted.

Reproduce (≈35 min per two variants on an 8-core laptop CPU):

```bash
python benchmarks/ieee_cis_benchmark.py --root data/raw/ieee-cis --variants none cumulative windowed
```
