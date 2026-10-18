import { useEffect, useState, type ReactNode } from "react";
import { api, type ReplayStatus } from "../lib/api";
import { ACTION_LABEL, RISK_LABEL, STATUS_LABEL, fmtInt, fmtProb } from "../lib/format";
import { usePublicConfig } from "../lib/config";

export function Card({ title, sub, actions, children, flush, className = "" }: {
  title?: ReactNode; sub?: ReactNode; actions?: ReactNode; children: ReactNode; flush?: boolean; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {sub && <p className="sub">{sub}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className={`card-body${flush ? " flush" : ""}`}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function PageHead({ title, desc, actions }: { title: string; desc?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {desc && <p>{desc}</p>}
      </div>
      {actions && <div className="row">{actions}</div>}
    </div>
  );
}

export const RiskBadge = ({ level }: { level: string }) => (
  <span className={`badge ${level}`}><span className="dot" />{RISK_LABEL[level] ?? level}</span>
);
export const ActionBadge = ({ action }: { action: string }) => (
  <span className={`badge ${action}`}><span className="dot" />{ACTION_LABEL[action] ?? action}</span>
);
export const StatusBadge = ({ status }: { status: string }) =>
  status === "none" ? <span className="muted">—</span> : (
    <span className={`badge ${status}`}><span className="dot" />{STATUS_LABEL[status] ?? status}</span>
  );
export const PsiBadge = ({ status }: { status: string }) => {
  const label: Record<string, string> = { ok: "Stable", warn: "Moderate shift", alert: "Significant shift", insufficient_data: "Collecting data" };
  return <span className={`badge ${status === "insufficient_data" ? "info" : status}`}><span className="dot" />{label[status] ?? status}</span>;
};
export const SyntheticTag = () => <span className="badge synthetic" title="Synthetic transaction (not real data)">Synthetic</span>;

export function ProbBar({ p, level }: { p: number; level: string }) {
  const w = Math.max(2, Math.min(100, (Math.log10(Math.max(p, 1e-4)) + 4) / 4 * 100)); // log scale, 0.01% .. 100%
  return (
    <div className="pbar" title={`${(p * 100).toFixed(3)}% (bar uses a log scale)`}>
      <div className="track"><div className={`fill ${level}`} style={{ width: `${w}%` }} /></div>
      <span className="v">{fmtProb(p)}</span>
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  return error ? <p className="err" role="alert">{error}</p> : null;
}

// ------------------------------------------------------------------------------ icons (inline, 18px)
const I = ({ d, size = 18 }: { d: string; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
);
export const Icon = {
  overview: () => <I d="M4 13h6V4H4v9Zm10 7h6V4h-6v16ZM4 20h6v-4H4v4Z" />,
  live: () => <I d="M3 12h4l3-8 4 16 3-8h4" />,
  cases: () => <I d="M9 12h6M9 16h6M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Zm7 0v5h5" />,
  model: () => <I d="M4 19V5m0 14h16M8 15l3-4 3 2 5-6" />,
  policy: () => <I d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4" />,
  monitor: () => <I d="M12 3a9 9 0 1 0 9 9M12 7v5l3 2M21 3l-6 6" />,
  play: () => <I d="M7 4v16l13-8L7 4Z" size={15} />,
  pause: () => <I d="M8 5v14M16 5v14" size={15} />,
  stop: () => <I d="M6 6h12v12H6z" size={15} />,
  moon: () => <I d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z" size={16} />,
  sun: () => <I d="M12 4V2m0 20v-2m8-8h2M2 12h2m13.7-5.7 1.4-1.4M4.9 19.1l1.4-1.4m0-11.4L4.9 4.9m14.2 14.2-1.4-1.4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" size={16} />,
  info: () => <I d="M12 8h.01M11 12h1v5h1M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z" size={16} />,
};

export function Shield() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 2 4 6.5v8.2c0 7.4 5.1 13.4 12 15.3 6.9-1.9 12-7.9 12-15.3V6.5L16 2Z" fill="var(--accent)" />
      <path d="m10.5 16.2 3.8 3.8 7.4-7.6" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ------------------------------------------------------------------------------ replay control
const SCENARIO_LABEL: Record<string, string> = {
  enriched: "Held-out data, fraud-enriched (≤5%)",
  historical: "Held-out data, natural order",
  synthetic_drift: "Synthetic drift scenario",
};

export function ReplayBar({ status, onChange }: { status: ReplayStatus | null; onChange: () => void }) {
  const [scenario, setScenario] = useState("enriched");
  const cfgScenarios = usePublicConfig().scenarios;
  const scenarios = Object.keys(cfgScenarios).length ? Object.keys(cfgScenarios) : Object.keys(SCENARIO_LABEL);
  const ieee = usePublicConfig().profile === "ieee";
  useEffect(() => { if (!scenarios.includes(scenario)) setScenario(scenarios[0]); }, [scenarios.join()]); // eslint-disable-line
  const [rate, setRate] = useState(25);
  const [count, setCount] = useState(3000);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const cfg = usePublicConfig();
  const maxRate = Math.min(200, cfg.max_replay_tps);
  const counts = [500, 1000, 3000, 10000, 50000].filter((n) => n <= cfg.max_replay_count);
  const running = status?.state === "running";
  const paused = status?.state === "paused";

  useEffect(() => { if (status?.rate_tps && (running || paused)) setRate(status.rate_tps); }, [status?.run_id]); // eslint-disable-line

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try { await fn(); onChange(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  const progress = status && status.target_count ? status.scored / status.target_count : 0;

  return (
    <div className="card" style={{ padding: "12px 16px", marginBottom: 16 }}>
      <div className="spread" style={{ flexWrap: "wrap" }}>
        <div className="row">
          <strong style={{ fontSize: 13.5 }}>Transaction replay</strong>
          {status && status.state !== "idle" && (
            <span className={`badge ${running ? "running" : status.state === "failed" ? "failed" : ""}`}>
              <span className="dot" />{status.state}
            </span>
          )}
          {status && status.state !== "idle" && (
            <span className="small ink2">
              {fmtInt(status.scored)} / {fmtInt(status.target_count)} scored · {fmtInt(status.alerts)} alerts
              · {status.effective_tps.toFixed(1)} tps{status.failed ? ` · ${status.failed} failed` : ""}
              {status.backpressure_waits ? ` · backpressure ${status.backpressure_waits}` : ""}
            </span>
          )}
        </div>
        <div className="row">
          <select className="select" value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={running || paused} aria-label="Scenario">
            {scenarios.map((k) => <option key={k} value={k}>{ieee && k === "historical" ? "Held-out stream, in order" : SCENARIO_LABEL[k] ?? k}</option>)}
          </select>
          <label className="row small ink2" style={{ gap: 6 }}>
            <span className="nowrap">{rate} tx/s</span>
            <input type="range" min={1} max={maxRate} value={Math.min(rate, maxRate)} style={{ width: 110 }}
              onChange={(e) => setRate(+e.target.value)}
              onPointerUp={() => (running || paused) && act(() => api.replayRate(rate))}
              onKeyUp={() => (running || paused) && act(() => api.replayRate(rate))} aria-label="Replay speed" />
          </label>
          <select className="select" value={count} onChange={(e) => setCount(+e.target.value)} disabled={running || paused} aria-label="Transactions">
            {counts.map((n) => <option key={n} value={n}>{fmtInt(n)} txns</option>)}
          </select>
          {!running && !paused && (
            <button className="btn primary" disabled={busy} onClick={() => act(() => api.replayStart({ scenario, rate_tps: rate, count, seed: 42 }))}>
              <Icon.play /> Start
            </button>
          )}
          {running && <button className="btn" disabled={busy} onClick={() => act(api.replayPause)}><Icon.pause /> Pause</button>}
          {paused && <button className="btn primary" disabled={busy} onClick={() => act(api.replayResume)}><Icon.play /> Resume</button>}
          {(running || paused) && <button className="btn" disabled={busy} onClick={() => act(api.replayStop)}><Icon.stop /> Stop</button>}
        </div>
      </div>
      {(running || paused) && (
        <div className="meter" style={{ marginTop: 10 }} aria-label="Replay progress"><div style={{ width: `${progress * 100}%` }} /></div>
      )}
      {status?.scenario && status.state !== "idle" && (
        <p className="note" style={{ marginTop: 8 }}>
          {status.scenario === "synthetic_drift" && "Synthetic transactions with an injected distribution shift — use to exercise drift monitoring. "}
          {status.scenario === "enriched" && "Real held-out transactions with every held-out fraud included at most once (up to 5%); alert rates are inflated by design. "}
          {status.scenario === "historical" && !ieee && "Real held-out transactions at the natural fraud rate (~0.13%). "}
          {status.scenario === "historical" && ieee && "Real held-out transactions at the natural fraud rate (~3.4%), in time order through the online feature store; each replay continues the stream. "}
          {ieee ? status.notes?.[1] : `Seed ${status.seed}.`} {status.data_source}
        </p>
      )}
      <ErrorNote error={error} />
    </div>
  );
}
