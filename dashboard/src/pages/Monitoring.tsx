import { api } from "../lib/api";
import { usePublicConfig } from "../lib/config";
import { fmtInt, fmtMs, fmtPct, fmtTime } from "../lib/format";
import { usePoll } from "../lib/hooks";
import { HBarChart, TimeSeriesChart } from "../components/charts";
import { Card, ErrorNote, PageHead, PsiBadge, Stat } from "../components/ui";

export default function Monitoring() {
  const cfg = usePublicConfig();
  const mon = usePoll(api.monitoring, 2000);
  const m = mon.data;
  const drift = m?.drift;
  const errors5 = m ? Object.values(m.errors_last_5m).reduce((a, b) => a + b, 0) : 0;
  const lp = m?.labelled_performance;

  return (
    <>
      <PageHead title="Monitoring"
        desc="Service health, input drift and score drift from a rolling window of the last 5,000 scored transactions. Fraud labels arrive late, so live accuracy is not shown as confirmed performance." />
      <ErrorNote error={mon.error} />
      <div className="kpis">
        <Stat label="HTTP scoring latency (60 s)" value={fmtMs(m?.request_latency_ms_last_60s.p50)}
          sub={m && m.request_latency_ms_last_60s.n === 0 ? "no HTTP scoring calls in the last 60 s (replay runs in-process)" : `p95 ${fmtMs(m?.request_latency_ms_last_60s.p95)} · p99 ${fmtMs(m?.request_latency_ms_last_60s.p99)}`} />
        <Stat label="Inference per transaction" value={fmtMs(m?.inference_latency_ms_per_txn.p50)} sub={`p95 ${fmtMs(m?.inference_latency_ms_per_txn.p95)}`} />
        <Stat label="Throughput" value={m ? `${m.throughput_tps_last_60s.toFixed(1)} tx/s` : "—"} sub="average over the last 60 s" />
        <Stat label="Errors (5 min)" value={fmtInt(errors5)} sub={m && Object.keys(m.errors_total).length ? Object.entries(m.errors_total).map(([k, v]) => `${k} ${v}`).join(" · ") : "none recorded"} />
        <Stat label="Alert rate (5 min)" value={fmtPct(m?.alert_rate_last_5m)} sub={drift?.expected_alert_rate != null ? `validation baseline ${fmtPct(drift.expected_alert_rate)}` : undefined} />
        <Stat label="Overall drift" value={drift?.overall_status ? <PsiBadge status={drift.overall_status} /> : "—"}
          sub={drift?.window_size != null ? `window ${fmtInt(drift.window_size)} transactions` : undefined} />
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Event throughput" sub="Scored transactions per second, last 2 minutes">
          {m && <TimeSeriesChart data={m.throughput_series} series={[{ key: "transactions", label: "Transactions", color: "var(--s1)", kind: "area" }]}
            xFormat={fmtTime} label="Transactions per second" />}
        </Card>
        <Card title="Prediction distribution" sub="PSI of the calibrated score vs. the validation period">
          {drift?.score ? (
            <div className="stack" style={{ gap: 10 }}>
              <div className="row"><PsiBadge status={drift.score.status} /><span className="ink2">PSI {drift.score.psi.toFixed(3)}</span></div>
              <dl className="kv">
                <dt>Mean probability (window)</dt><dd>{fmtPct(drift.score.mean, 3)}</dd>
                <dt>Mean probability (validation)</dt><dd>{fmtPct(drift.score.reference_mean, 3)}</dd>
                <dt>Alert rate (5 min)</dt><dd>{fmtPct(m?.alert_rate_last_5m)}</dd>
              </dl>
              {cfg.profile !== "ieee" && <p className="note">The fraud-enriched replay raises the score distribution by design, so this monitor should flag it.</p>}
            </div>
          ) : <div className="empty">No scored transactions yet.</div>}
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Feature drift" sub={drift?.reliable === false ? `Collecting data — PSI needs ≥ 500 transactions (have ${fmtInt(drift?.window_size)})` : "Population Stability Index vs. training distribution (top 12)"}>
          {drift?.features?.length ? (
            <>
              <HBarChart label="Feature PSI" labelWidth={100} valueFormat={(v) => v.toFixed(3)}
                rows={drift.features.slice(0, 12).map((f) => ({ label: f.seasonal ? `${f.feature} *` : f.feature, value: f.psi, detail: f.seasonal ? "seasonal: excluded from overall status" : f.status }))} color="var(--s1)" />
              <p className="note">Bands: &lt; 0.10 stable · 0.10–0.25 moderate · ≥ 0.25 significant (industry rule of thumb, not a significance test).
                {" "}{drift.n_features_alert ?? 0} significant, {drift.n_features_warn ?? 0} moderate. * Hour-of-day features shift whenever the window covers part of a day, so they are shown but not used for the overall status.</p>
            </>
          ) : <div className="empty">No data yet.</div>}
        </Card>
        <div className="stack">
          <Card title="Data-quality warnings" sub="Inputs outside the range seen in training (model behaviour there is unvalidated)">
            {m && Object.keys(m.data_quality_warnings).length ? (
              <table className="data"><tbody>{Object.entries(m.data_quality_warnings).map(([k, v]) => (
                <tr key={k}><td className="mono">{k}</td><td className="num">{fmtInt(v)}</td></tr>))}</tbody></table>
            ) : <p className="muted" style={{ margin: 0 }}>No warnings recorded.</p>}
          </Card>
          <Card title="Outcomes with known labels" sub={lp?.note}>
            {lp?.available ? (
              <dl className="kv">
                <dt>Replayed labelled transactions</dt><dd>{fmtInt(lp.n)} ({lp.frauds} frauds)</dd>
                <dt>Alert precision</dt><dd>{fmtPct(lp.precision)} <span className="muted small">({lp.true_positives} of {fmtInt(lp.alerts)} alerts)</span></dd>
                <dt>Fraud recall</dt><dd>{fmtPct(lp.recall)}</dd>
                <dt>Mean predicted vs observed</dt><dd>{fmtPct(lp.mean_predicted, 3)} vs {fmtPct(lp.observed_rate, 3)}</dd>
                <dt>Analyst decisions</dt><dd>{fmtInt(m?.analyst_feedback.confirmed_fraud ?? 0)} fraud · {fmtInt(m?.analyst_feedback.legitimate ?? 0)} legitimate</dd>
              </dl>
            ) : <p className="muted" style={{ margin: 0 }}>No labelled outcomes yet. Replay held-out history to simulate delayed labels.</p>}
          </Card>
          <p className="note">Prometheus metrics: <a href="/metrics" target="_blank" rel="noreferrer">/metrics</a> · Liveness <a href="/health">/health</a> · Readiness <a href="/ready">/ready</a> · {m?.stream_subscribers ?? 0} live stream subscribers.</p>
        </div>
      </div>
    </>
  );
}
