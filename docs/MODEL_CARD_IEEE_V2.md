# Model card — FraudShield v2 (IEEE-CIS, authorisation-time fields + velocity)

| | |
|---|---|
| **Model version** | `ieee-v2-raw-plus-velocity-20261017-171440` |
| **Status** | **Failed the pre-registered promotion gate**; served locally only through a logged `--force` override, for demonstration |
| **Type** | XGBoost (438 trees, depth 8) on 43 inputs + Platt calibration |
| **Inputs** | 24 raw authorisation-time fields, 4 row-local derived inputs, 15 velocity features from the online feature store |
| **Output** | Calibrated fraud probability; a separate policy maps it to approve / review / decline; alerts carry plain-language risk factors |
| **Serving profile** | `FRAUDSHIELD_PROFILE=ieee` (single instance: the feature store is in process) |
| **Owner / contact** | Armand Junior Dongmo Notue (portfolio project) |
| **Artifact** | `artifacts/ieee/models/<version>/` with SHA-256 manifest; never committed or hosted (competition data) |

## Intended use

* Demonstrating a merchant-side real-time fraud system: velocity features computed online with
  proven parity to training, linked activity for investigators, monthly backtesting.
* Research on how much of the vendor-feature performance a merchant can recover from its own stream.

## Out of scope

* Any production decision. The model failed its own quality gate (below).
* Use outside the IEEE-CIS population (one e-commerce dataset, about 6 months, provided by Vesta via
  a Kaggle competition). The competition rules do not allow redistributing the data.
* Decisions about individuals without human review.

## Training data

IEEE-CIS `train_transaction` + `train_identity`, field contract only (no vendor `C`/`D`/`V`/`id_`
columns). Chronological split on `TransactionDT`: train = first 60% (354,324 transactions, 3.38% fraud),
validation = next 20% (3.90%), test = final 20% (118,108 transactions, 4,064 frauds, 3.44%).
Categories learned on train only; unseen or missing values encoded as missing.

## Evaluation (test window, scored once)

| Metric | Raw contract | **Raw + velocity (this model)** |
|---|---:|---:|
| PR-AUC [95% CI] | 0.208 [0.196, 0.222] | **0.217** [0.204, 0.231] |
| Paired Δ PR-AUC [95% CI] | — | +0.009 [+0.001, +0.018] |
| ROC-AUC | 0.806 | 0.821 |
| Recall at 1% FPR | 16.2% | 17.3% |
| Precision in the top 0.1% of scores | 57.6% | 65.3% |

* Reference: the all-431-column model reaches PR-AUC 0.508. Velocity recovers 3% of that gap.
* At the review threshold 0.167 (amended rule, post-hoc on test): 4.1% of transactions reviewed,
  precision 26.3%, recall 31.1%.
* Calibration: ECE 0.0035, Brier 0.0300.
* Latency (in-process, laptop): p50 0.79 ms, p95 1.35 ms including the feature-store read + update.
* Parity: 0 mismatches between online and offline velocity features (1,771,620 values on the full
  test window; 75,000 values through the running API).

Full report: [ML_EVALUATION.md §12](ML_EVALUATION.md#12-fraudshield-v2-ieee-cis-on-authorisation-time-fields--online-velocity-features).

## Promotion gate

Declared before any v2 result: test PR-AUC ≥ 0.30, ROC-AUC ≥ 0.85, p95 latency ≤ 50 ms. Result:
PR-AUC 0.217 **fail**, ROC-AUC 0.821 **fail**, latency 1.35 ms pass. The limits were not changed.

## Factors and limitations

* **Weak absolute performance.** Most frauds are not separable from legitimate traffic with these
  fields; at a 1% false-positive rate the model catches 17% of fraud.
* **No true card or account key.** `card1`–`card5` pools several real cards, which dilutes "per card"
  velocity. `DeviceInfo` is a device model shared by many customers, so device links are weak.
* **Decay.** In the monthly backtest a model trained once falls from 0.253 to 0.198 PR-AUC in three
  months. Monthly retraining (one month of label-maturity gap) holds 0.24–0.25.
* **Event order.** Online features assume events arrive in time order. Late events are processed at
  the stream watermark and flagged, so their features can differ from an offline recomputation.
* **Fairness** cannot be assessed: the data has no customer attributes.
* **Explanations** are TreeSHAP attributions rendered as sentences ("14 earlier transactions on this
  card in the last 24 h (typical: 6)"). They describe model associations, not causes or evidence.

## Monitoring and maintenance

The same live monitoring as the credit-card model applies: input and score PSI (with a separate
missing-value bin, since many IEEE-CIS fields are often empty), alert rate, unseen-category and
out-of-range warnings, late-event counts, latency and errors.

* **Retraining:** monthly, on labels at least one month old.
* **Release:** promotion through the evaluation gate. This version would not pass it.
