---
title: FraudShield AI
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: Real-time fraud scoring, decisions, SHAP and drift
---

# FraudShield AI — public demo

Live demo of [FraudShield AI](https://github.com/ARMAND-cod-eng/fraud-detection-ensemble): a
real-time card-fraud scoring service with calibrated probabilities, a versioned decision policy,
per-decision TreeSHAP explanations, an analyst investigation workspace and drift monitoring.

**Try it:** on the Overview page, start a replay of held-out transactions, open an alert under
*Investigations*, then run the *Synthetic drift* scenario and watch *Monitoring*. The API is
documented at `/docs`.

**How this instance was built:** the Space image downloads the public dataset (SHA-256 verified)
and trains the model with the repository's pipeline (`configs/training_fast.yaml`: the same
out-of-time protocol with a smaller search). The full benchmark numbers are in the GitHub README.

**Demo limits:** replays are capped at 100 transactions/second and 5,000 transactions; policy
changes and data resets are disabled; the instance is shared by all visitors and its data is
ephemeral.

**Data:** Credit Card Fraud Detection, Machine Learning Group — ULB and Worldline (September 2013,
European cardholders), made available under the Open Database License 1.0 (database) and the
Database Contents License 1.0 (contents): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
Code: MIT.
