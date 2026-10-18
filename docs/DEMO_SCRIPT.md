# 90-second demo script

**Setup (before recording):** model trained and promoted, `uvicorn api.main:app` running, dashboard
built, `POST /api/v1/admin/reset` called, browser at `http://localhost:8000/#/overview`.

| Time | Screen | Action | Say |
|---|---|---|---|
| 0:00–0:10 | Overview | — | "FraudShield scores card transactions in about a millisecond, turns calibrated probabilities into decisions, and gives analysts the evidence to act. Everything you'll see is measured on held-out data." |
| 0:10–0:22 | Overview → replay bar | Scenario **Held-out data, fraud-enriched**, 60 tx/s, Start | "This replays real transactions from the last 20% of the timeline — data the model never saw — through the same API path as production traffic. This scenario enriches the stream with every real held-out fraud (each once) so the demo is eventful." |
| 0:22–0:35 | Live monitor | Let rows stream; filter **Alerts** | "Each row is scored, routed by a versioned policy, persisted and streamed over server-sent events. Note the inference time column — under a millisecond." |
| 0:35–0:55 | Investigations | Open the top alert | "For an alert the analyst sees the calibrated probability, why the policy recommended decline, unsupervised anomaly signals as context, and a TreeSHAP breakdown with training percentiles — clearly labelled as model attributions, not proof of fraud." Click **Confirm fraud** with a note. "Decisions are stored with an audit trail; for replayed history the true label is revealed only after the decision." |
| 0:55–1:10 | Model performance | Point at the table | "The original project claimed an ensemble. I re-ran it faithfully, fixed leakage, and benchmarked out-of-time with confidence intervals: the ensemble adds +0.006 PR-AUC with an interval that includes zero, at 30–40× the latency — so the pre-registered rule ships XGBoost and keeps the detectors as analyst context." |
| 1:10–1:22 | Decision policy | Drag the review threshold down | "Thresholds are business policy, not model code. The what-if shows frauds stopped versus review workload and cost, under assumptions that are stated, not hidden." |
| 1:22–1:30 | Monitoring | Start **Synthetic drift** replay | "Drift monitoring compares live inputs and scores with the training distribution; the synthetic shift lights it up within seconds. Everything is tested, containerised and reproducible from one command." |

**If asked "why not deep learning / a GNN?":** the autoencoder is in the benchmark (test PR-AUC
0.21 out-of-time) and the dataset has no entity identifiers, so a graph model would be invented
structure. On IEEE-CIS, which has card/address/device fields, past-only entity counts were tested:
cumulative counts drifted and hurt (−0.010 PR-AUC), trailing windows fixed the drift but added nothing.

**If asked "how would this work in real time with velocity features?" (v2):** "Every velocity
feature is declared once and computed by two engines: a vectorised one for training and a
per-entity online store for serving. A transaction reads its features as of before it happened,
then updates the store, and retries never double-count. I proved parity on 118k real transactions,
with 0 mismatches in 1.77M values, and again through the running API. The honest result: on
authorisation-time fields the features add +0.009 PR-AUC, real but small. A static model loses 22%
in three months, so I'd retrain monthly. And the model failed the gate I set before the run, so it
doesn't ship."
