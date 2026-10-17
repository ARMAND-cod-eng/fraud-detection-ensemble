// Typed client for the FraudShield API (same origin in production, proxied by Vite in dev).

export type Action = "approve" | "review" | "decline";
export type Risk = "low" | "medium" | "high";
export type ReviewStatus = "none" | "open" | "confirmed_fraud" | "legitimate" | "escalated";

export interface TxnSummary {
  transaction_id: string;
  received_at: number;
  event_time: number | null;
  amount: number;
  fraud_probability: number;
  risk_level: Risk;
  action: Action;
  reasons?: string[];
  model_version?: string;
  policy_version?: string;
  latency_ms: number;
  source: string;
  synthetic: boolean;
  scenario: string | null;
  review_status: ReviewStatus;
  over_capacity?: boolean;
  has_explanation?: boolean;
  quality?: string[];
}

export interface Factor {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
  training_percentile: number | null;
  description: string;
}

export interface Explanation {
  method: string;
  space: string;
  base_value: number;
  output_value: number;
  top_factors: Factor[];
  other_features_contribution: number;
  disclaimer: string;
}

export interface Signal {
  value: number;
  percentile_vs_normal: number | null;
  used_in_score: boolean;
  description: string;
}

export interface CaseEvent {
  id: number;
  at: number;
  actor: string;
  event: string;
  decision: string | null;
  notes: string | null;
}

export interface TxnDetail extends TxnSummary {
  raw_score: number;
  expected_loss: number;
  features: Record<string, number>;
  explanation: Explanation | null;
  signals: Record<string, Signal> | null;
  quality: string[];
  history: CaseEvent[];
  ground_truth_revealed: number | null;
}

export interface Percentiles { p50: number | null; p95: number | null; p99?: number | null; mean: number | null; n: number }

export interface ReplayStatus {
  state: "idle" | "running" | "paused" | "completed" | "stopped" | "failed";
  run_id: string | null;
  scenario: string | null;
  seed: number | null;
  rate_tps: number;
  target_count: number;
  produced: number;
  scored: number;
  failed: number;
  alerts: number;
  backpressure_waits: number;
  queue_depth: number;
  elapsed_s: number;
  effective_tps: number;
  last_error: string | null;
  data_source: string | null;
  notes: string[];
}

export interface Overview {
  stats: {
    n: number; alerts: number; reviews: number; declines: number; volume: number;
    expected_fraud_amount: number; flagged_amount: number; open_cases: number; confirmed_fraud: number;
    confirmed_legit: number; synthetic: number; alert_rate: number;
    risk_distribution: Record<string, number>;
    probability_histogram: { bin: string; count: number }[];
  };
  latency: { request_ms: Percentiles; inference_ms: Percentiles };
  throughput_tps_last_60s: number;
  activity: { t: number; transactions: number; alerts: number; avg_latency_ms: number | null }[];
  model: { version: string; kind: string; status: string; loaded_at: number };
  policy_version: string;
  replay: ReplayStatus | null;
  exposure_note: string;
}

export interface DriftFeature { feature: string; psi: number; status: "ok" | "warn" | "alert"; seasonal?: boolean }
export interface Monitoring {
  uptime_s: number;
  request_latency_ms: Percentiles;
  request_latency_ms_last_60s: Percentiles;
  inference_latency_ms_per_txn: Percentiles;
  throughput_tps_last_60s: number;
  alert_rate_last_5m: number | null;
  errors_last_5m: Record<string, number>;
  errors_total: Record<string, number>;
  data_quality_warnings: Record<string, number>;
  drift: {
    available: boolean; window_size?: number; reliable?: boolean; overall_status?: string;
    features?: DriftFeature[]; score?: { psi: number; status: string; mean: number; reference_mean: number };
    max_feature_psi?: number; n_features_warn?: number; n_features_alert?: number; expected_alert_rate?: number;
  };
  throughput_series: { t: number; transactions: number; alerts: number }[];
  labelled_performance: {
    available: boolean; n: number; note: string; frauds?: number; alerts?: number; true_positives?: number;
    precision?: number | null; recall?: number | null; mean_predicted?: number; observed_rate?: number;
  };
  analyst_feedback: Record<string, number>;
  stream_subscribers: number;
}

export interface PolicyConfig {
  policy_id: string; description: string; review_threshold: number; decline_threshold: number;
  expected_loss_review: boolean; max_reviews_per_hour: number;
  cost_assumptions: { review_cost: number; loss_rate: number; false_decline_cost: number };
}
export interface PolicyState {
  version: string; config: PolicyConfig; model_review_threshold: number | null; reviews_last_hour: number;
  history: { version: string; at: number; actor: string }[]; disclaimer: string;
}
export interface PolicyInput {
  review_threshold: number; decline_threshold: number; expected_loss_review: boolean;
  max_reviews_per_hour: number; review_cost?: number; false_decline_cost?: number;
}
export interface Simulation {
  data: string; window_hours: number;
  result: {
    transactions: number; approved: number; reviewed: number; declined: number; frauds: number; frauds_stopped: number;
    fraud_amount_total: number; fraud_amount_stopped: number; legit_declined: number; legit_reviewed: number;
    cost: { fraud_loss: number; review: number; false_declines: number; total: number };
  };
  capacity: { reviews_per_hour: number; reviews: number; frauds_caught: number; recall: number; precision: number; residual_fraud_loss: number }[];
  assumptions: Record<string, number>;
}

// The evaluation report is large; only the parts the UI reads are typed.
export interface ThresholdMetrics { threshold: number; precision: number; recall: number; f1: number; fpr: number; alerts: number; alert_rate: number; tp: number; fp: number; fn: number; tn: number }
export interface CI { point: number; ci_low: number; ci_high: number; delta_vs_xgboost?: number; delta_ci_low?: number; delta_ci_high?: number; p_delta_le_0?: number }
export interface Evaluation {
  version: string; experiment: string; created_at: string; selected_model: string;
  dataset: { sha256: string; rows_raw: number; frauds_raw: number; rows: number; frauds: number; exact_duplicates_removed: number; time_span_hours: number;
    split: Record<string, { rows: number; frauds: number; fraud_rate: number; time_start_s: number; time_end_s: number }> & { strategy: string } };
  protocol: Record<string, unknown> & { selection_log: string[] };
  candidates: Record<string, { family: string; notes: string; info: Record<string, unknown> }>;
  validation: Record<string, Record<string, number>>;
  test: Record<string, Record<string, number> & { at_val_f1_threshold: ThresholdMetrics }>;
  test_bootstrap_95ci: { pr_auc: Record<string, CI>; roc_auc: Record<string, CI> };
  selected_operating_points: {
    review_threshold: number; review_threshold_test: ThresholdMetrics; review_threshold_test_recall_ci: CI;
    review_threshold_test_precision_ci: CI; f1_threshold: number; f1_threshold_test: ThresholdMetrics;
  };
  calibration_test: Record<"raw" | "calibrated", { brier: number; log_loss: number; ece: number;
    reliability: { lo: number; hi: number; n: number; mean_predicted: number; observed_rate: number; positives: number }[] }>;
  cost_experiment: {
    assumptions: Record<string, number | string>; validation_cost_optimal_threshold: number;
    test: Record<string, { total_cost: number; reviews?: number; frauds_caught?: number; fraud_loss?: number; review_cost?: number; threshold?: number }>;
    validation_curve: { threshold: number; total_cost: number; reviews: number }[];
    test_curve: { threshold: number; total_cost: number; reviews: number }[];
  };
  latency_in_process: Record<string, { single_row_p50_ms: number; single_row_p95_ms: number; single_row_p99_ms: number; batch_rows_per_s: number }>;
  global_importance: { feature: string; mean_abs_shap: number }[];
  curves_test: Record<string, { roc: [number, number][]; pr: [number, number][] }>;
  environment: Record<string, string | null>;
  runtime_seconds: number | null;
}
export interface ModelInfo {
  model_version: string; kind: string; selected_candidate: string; created_at: string; data_sha256: string;
  uses_anomaly_signals_in_score: boolean; auxiliary_signals_available: boolean;
  thresholds: { review: number; f1: number }; calibration: { method: string; a: number; b: number };
}
export interface Versions { production: string | null; versions: string[]; serving: string | null;
  history: { action: string; version: string; previous: string | null; reason: string; at: string }[] }

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const text = await res.text();
  const body = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const err = body?.error ?? {};
    throw new ApiError(res.status, err.code ?? "http_error", err.message ?? `HTTP ${res.status}`);
  }
  return body as T;
}

const post = <T>(path: string, body?: unknown) => request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  overview: () => request<Overview>("/api/v1/overview"),
  monitoring: () => request<Monitoring>("/api/v1/monitoring"),
  transactions: (q: Record<string, string | number | boolean | undefined>) => {
    const params = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => { if (v !== undefined && v !== "") params.set(k, String(v)); });
    return request<{ items: TxnSummary[]; total: number }>(`/api/v1/transactions?${params}`);
  },
  transaction: (id: string) => request<TxnDetail>(`/api/v1/transactions/${encodeURIComponent(id)}`),
  explanation: (id: string) => request<{ explanation: Explanation; signals: Record<string, Signal> }>(`/api/v1/transactions/${encodeURIComponent(id)}/explanation`),
  review: (id: string, decision: string, notes?: string) => post<CaseEvent>(`/api/v1/transactions/${encodeURIComponent(id)}/review`, { decision, notes: notes || null }),
  modelInfo: () => request<ModelInfo>("/api/v1/models/info"),
  evaluation: () => request<Evaluation>("/api/v1/models/evaluation"),
  versions: () => request<Versions>("/api/v1/models/versions"),
  policy: () => request<PolicyState>("/api/v1/policy"),
  updatePolicy: (p: PolicyInput) => request<{ version: string }>("/api/v1/policy", { method: "PUT", body: JSON.stringify(p) }),
  simulate: (p: PolicyInput) => post<Simulation>("/api/v1/policy/simulate", p),
  replayStatus: () => request<ReplayStatus>("/api/v1/replay/status"),
  replayScenarios: () => request<Record<string, string>>("/api/v1/replay/scenarios"),
  replayStart: (b: { scenario: string; rate_tps: number; count: number; seed: number }) => post<ReplayStatus>("/api/v1/replay/start", b),
  replayPause: () => post<ReplayStatus>("/api/v1/replay/pause"),
  replayResume: () => post<ReplayStatus>("/api/v1/replay/resume"),
  replayStop: () => post<ReplayStatus>("/api/v1/replay/stop"),
  replayRate: (rate_tps: number) => request<ReplayStatus>("/api/v1/replay", { method: "PATCH", body: JSON.stringify({ rate_tps }) }),
  reset: () => post<{ status: string }>("/api/v1/admin/reset"),
};
