const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const eur = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 2 });
const eur0 = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const fmtInt = (n: number | null | undefined) => (n == null ? "—" : nf0.format(n));
export const fmtCompact = (n: number | null | undefined) => (n == null ? "—" : n < 10_000 ? nf0.format(n) : compact.format(n));
export const fmtEur = (n: number | null | undefined, whole = false) => (n == null ? "—" : (whole ? eur0 : eur).format(n));

export function fmtPct(p: number | null | undefined, digits?: number): string {
  if (p == null || Number.isNaN(p)) return "—";
  const v = p * 100;
  const d = digits ?? (v !== 0 && Math.abs(v) < 1 ? 2 : 1);
  return `${v.toFixed(d)}%`;
}

export function fmtProb(p: number): string {
  if (p >= 0.995) return ">99%";
  if (p < 0.0001) return "<0.01%";
  return fmtPct(p);
}

export const fmtMs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : ms < 100 ? `${ms.toFixed(1)} ms` : `${ms.toFixed(0)} ms`;

export const fmtNum = (n: number | null | undefined, d = 3) => (n == null ? "—" : n.toFixed(d));

export function fmtTime(epochS: number): string {
  return new Date(epochS * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function fmtDateTime(epochS: number): string {
  return new Date(epochS * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
}

export function fmtAgo(epochS: number): string {
  const s = Math.max(0, Date.now() / 1000 - epochS);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

/** Dataset "Time" is seconds since the start of the 48 h capture; show it as day + clock. */
export function fmtEventTime(sec: number | null): string {
  if (sec == null) return "—";
  const day = Math.floor(sec / 86400) + 1;
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `Day ${day} · ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export const ACTION_LABEL: Record<string, string> = { approve: "Approve", review: "Review", decline: "Decline" };
export const RISK_LABEL: Record<string, string> = { low: "Low", medium: "Medium", high: "High" };
export const STATUS_LABEL: Record<string, string> = {
  none: "—", open: "Open", confirmed_fraud: "Confirmed fraud", legitimate: "Legitimate", escalated: "Escalated",
};
export const MODEL_LABEL: Record<string, string> = {
  xgboost: "XGBoost", stacked: "Stacked ensemble (OOF)", stacked_insample: "Stacked (legacy, in-sample)",
  logreg: "Logistic regression", isolation_forest: "Isolation Forest", autoencoder: "Autoencoder",
  xgboost_raw_features: "XGBoost (raw features)",
};
