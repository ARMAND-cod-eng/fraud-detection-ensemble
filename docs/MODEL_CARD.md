# Model card — FraudShield credit-card fraud scorer

| | |
|---|---|
| **Model version** | `xgboost-20261016-204043` (registry production) |
| **Type** | Gradient-boosted trees (XGBoost 3.4.1, 103 trees, depth 5) + Platt calibration |
| **Output** | Calibrated probability that a transaction is fraudulent; a separate policy maps it to approve / review / decline |
| **Auxiliary signals** | Isolation Forest and autoencoder anomaly percentiles, shown to analysts as context; **not** used in the score |
| **Owner / contact** | Armand Junior Dongmo Notue (portfolio project) |
| **Artifact** | `artifacts/models/<version>/` with SHA-256 manifest, metadata, evaluation and drift reference |

## Intended use

* Demonstration and research of a fraud-scoring workflow: real-time scoring, decision policy,
  investigation and monitoring on a public benchmark dataset.
* Ranking transactions for **human review**. Decline recommendations are suggestions under an
  illustrative policy.

## Out of scope

* Production fraud decisions for any real institution. The model was trained on two days of
  2013 European card data with anonymised features; it has not been validated on any other
  population, period, channel or currency.
* Automated adverse decisions about individuals without human oversight.
* Any use that requires knowing *why* in business terms: V1–V28 are undisclosed PCA components.

## Training data

ULB credit-card dataset (mlg-ulb/creditcardfraud, Database Contents License), first 60% of the
timeline after removing 1,081 exact duplicates: 170,235 transactions, 342 frauds (0.20%).
Inputs: V1–V28, Amount and 7 row-local engineered features (log amount, cyclical hour assuming
`Time=0` is midnight, norm/max/mean/std of the PCA vector). Raw `Time` is not an input.

## Evaluation

Out-of-time: validation = next 20% (57 frauds) for every choice; test = final 20% (74 frauds),
scored once. Full report: [ML_EVALUATION.md](ML_EVALUATION.md).

| Test metric | Value [95% CI] |
|---|---|
| PR-AUC | 0.801 [0.713, 0.881] |
| ROC-AUC | 0.974 [0.954, 0.991] |
| Recall at 0.1% false-positive rate | 81.1% |
| At review threshold 0.533 (validation cost-optimal) | precision 94.5% [87.8, 100] · recall 70.3% [60.6, 80.3] · 55 alerts / 56,746 |
| Brier / ECE (calibrated) | 0.000394 / 0.00020 |
| Single-transaction latency (in-process, p95) | 0.84 ms |

Baselines on the same split: logistic regression PR-AUC 0.789, stacked ensemble 0.808 (difference
vs XGBoost within noise), XGBoost without engineered features 0.791.

## Factors and limitations

* **Small positive counts.** Metrics move several points per fraud; read every number with its CI.
* **Short horizon.** Two days of data; the test window is ~7.7 h. Long-term drift, seasonality and
  adversarial adaptation are untested.
* **Missed frauds are mostly low-score.** ~15% of test frauds are not reached even at a 1% false-positive rate; their scores are indistinguishable
  from normal traffic; lower thresholds add reviews, not catches.
* **Small amounts.** Median test fraud is €4.79; cost-only policies would ignore them, although
  small frauds can be card-testing that precedes larger fraud.
* **No subgroup analysis.** The data has no customer attributes, so fairness across groups cannot
  be assessed; any deployment on real data must add it.
* **Explanations** are TreeSHAP attributions of this model on anonymised components: statistical
  associations, not causes or evidence.

## Ethical considerations

False positives inconvenience legitimate customers (declines) and consume analyst time; false
negatives cost money and harm cardholders. The project therefore keeps model and policy separate,
records the model and policy version on every decision, routes uncertain cases to humans,
reveals delayed ground truth only after an analyst decides, and documents all cost assumptions.

## Monitoring and maintenance

Live PSI drift on inputs (training reference) and scores (validation reference), alert-rate
shift, data-quality warnings for out-of-range inputs, latency and error metrics. Retraining
produces a new immutable version; promotion requires the evaluation gate; rollback is one
command (`fraudshield models rollback`).
