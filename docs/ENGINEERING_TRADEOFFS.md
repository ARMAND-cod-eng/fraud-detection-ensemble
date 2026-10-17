# Engineering trade-offs

Each decision below names the alternative that was considered and why it was not chosen for
this release. "Simplest credible architecture" was the tie-breaker.

## Machine learning

| Decision | Alternative | Why |
|---|---|---|
| **Serve plain XGBoost, keep the anomaly detectors as analyst context** | Serve the stacked ensemble (the original design) | Pre-registered rule on validation: the ensemble's PR-AUC gain (+0.0015) was below 0.01 and its CI included 0; on test the gain is +0.006 with CI [−0.003, +0.018]. It costs 30–40× the latency (p95 25–40 ms across two runs vs 0.84 ms) and adds a pickle-based component to the hot path. The detectors still run on demand for investigations. |
| **Chronological split** | Random stratified split (legacy) | Fraud models are deployed forward in time. Same pipeline on a random split reports test PR-AUC 0.872 vs 0.801 out-of-time — the random protocol is optimistic. |
| **Drop exact duplicates before splitting** | Keep them | 1,081 duplicates; with a random split 260–298 test rows had an exact copy in training. |
| **PR-AUC as the selection metric** | ROC-AUC / F1 @ 0.5 | With 0.1–0.2% positives ROC-AUC is dominated by the easy negatives; F1 depends on an arbitrary threshold. Ranking quality in the top alerts is what an investigation team consumes. |
| **Platt scaling, chosen a priori** | Isotonic; choose by test calibration | 57 validation frauds is too few for a stable isotonic map; choosing the method on test would be tuning on test. Here the selected XGBoost (no class re-weighting) was already well calibrated (ECE 0.0002), so calibration is a safeguard that matters more for re-weighted models. |
| **Native TreeSHAP (`pred_contribs`)** | `shap` package | Same algorithm for tree models, exact additivity, no numba/compiled dependency in the serving image. |
| **Autoencoder trained in PyTorch, served in NumPy** | Ship PyTorch / ONNX Runtime in the API | A 5-layer MLP is ~20 lines of NumPy; a unit test proves both forward passes agree. Keeps the API image free of a ~700 MB dependency. |
| **Out-of-fold detector scores for stacking** | In-sample scores (legacy) | In-sample autoencoder errors are label-dependent (the AE saw only training normals). Kept a `stacked_insample` reference model to show the effect (small here, real by construction). |
| **Bootstrap CIs and paired deltas** | Point estimates | 57–74 frauds per split: a single fraud moves recall by 1.4 points. Paired resampling answers "is model A better than B on the same data?" |

## Serving and product

| Decision | Alternative | Why |
|---|---|---|
| **One FastAPI process (1 worker) + SQLite WAL** | Postgres + Redis + several workers | Zero external services for a reviewer to run. Replay state, SSE fan-out, rate-limit buckets and drift windows are in-process, so the service runs with **one worker by design**. Horizontal scaling would move them to Redis (pub/sub, token buckets) and Postgres — the store interface is intentionally narrow to make that swap local. |
| **No Kafka** | Kafka/Redpanda for the transaction stream | The replay engine already demonstrates pacing, bounded buffering, backpressure and failure isolation. A broker adds an always-on service without changing what the demo proves. It becomes worthwhile with multiple producers/consumers or durable replay. |
| **Server-sent events** | WebSockets | Updates are one-directional; SSE is plain HTTP, reconnects automatically and passes through proxies. |
| **Explanations computed eagerly only for alerts** | For every transaction / never | Analysts open alerts; approvals rarely need one. Keeps the approval path at model latency; approvals are explained lazily and cached. |
| **Idempotency keyed on `transaction_id` + payload hash** | `Idempotency-Key` header | The transaction ID is the natural key in payments; a hash mismatch is a client bug worth a 409. |
| **Policy separate from the model, versioned** | Thresholds baked into the model | Business owners change thresholds and costs without retraining; every decision is traceable to `model_version` + `policy_version`. |
| **React + TypeScript with hand-written SVG charts** | Chart/UI libraries | 88 kB gzipped total, full control over accessibility and the visual system, no dependency churn. Cost: more code to maintain. |
| **File-based model registry** | MLflow Model Registry | Promotion/rollback must work with zero services; MLflow tracking is optional (`pip install -e .[tracking]`) for experiment history. |
| **Hash-based artifact integrity** | Signed artifacts (Sigstore/KMS) | Checksums catch corruption and tampering of a single file; signing is the production step (see DEPLOYMENT). |

## Things deliberately not built

* Online learning / automatic retraining: unsafe without delayed-label handling and an approval step.
* GNNs: the ULB data has no entity identifiers; inventing relationships would produce meaningless
  "graph" gains. Past-only entity features are implemented for IEEE-CIS, which does have them.
* Grafana alert rules and PagerDuty routing: the dashboards are provisioned, alerting is environment-specific.
