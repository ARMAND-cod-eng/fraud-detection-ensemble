import { api } from "../lib/api";
import { usePublicConfig } from "../lib/config";
import { fmtCompact, fmtMoney, fmtInt, fmtMs, fmtPct, fmtProb, fmtTime } from "../lib/format";
import { usePoll } from "../lib/hooks";
import { BarChart, TimeSeriesChart } from "../components/charts";
import { ActionBadge, Card, ErrorNote, PageHead, ReplayBar, RiskBadge, Stat, SyntheticTag } from "../components/ui";

export default function Overview() {
  const cfg = usePublicConfig();
  const ov = usePoll(api.overview, 2000);
  const alerts = usePoll(() => api.transactions({ alerts_only: true, limit: 8 }), 3000);
  const d = ov.data;
  const s = d?.stats;
  // Trim leading empty intervals so a fresh session does not show 10 minutes of zeros.
  const first = d ? d.activity.findIndex((a) => a.transactions > 0) : -1;
  const activity = d ? d.activity.slice(first < 0 ? d.activity.length - 12 : Math.max(0, Math.min(first - 1, d.activity.length - 12))) : [];

  return (
    <>
      <PageHead
        title="Executive overview"
        desc={`Live view of everything the scoring service has processed in this session. Probabilities are calibrated model estimates; amounts are in ${cfg.currency}.`}
      />
      <ReplayBar status={d?.replay ?? null} onChange={ov.refresh} />
      <ErrorNote error={ov.error} />

      <div className="kpis">
        <Stat label="Transactions processed" value={fmtCompact(s?.n)} sub={s?.synthetic ? `${fmtInt(s.synthetic)} synthetic` : "All real held-out data"} />
        <Stat label="Fraud alerts" value={fmtInt(s?.alerts)} sub={s ? `${fmtInt(s.reviews)} review · ${fmtInt(s.declines)} decline` : undefined} />
        <Stat label="Alert rate" value={s ? fmtPct(s.alert_rate) : "—"} sub={s ? `${fmtInt(s.open_cases)} open cases` : undefined} />
        <Stat label="Expected fraud exposure" value={s ? fmtMoney(s.expected_fraud_amount, true) : "—"}
          sub={<span title={d?.exposure_note}>Σ probability × amount (model estimate)</span>} />
        <Stat label="Inference latency" value={fmtMs(d?.latency.inference_ms.p50)}
          sub={`p95 ${fmtMs(d?.latency.inference_ms.p95)} · request p95 ${fmtMs(d?.latency.request_ms.p95)}`} />
        <Stat label="Model" value={<span style={{ fontSize: 17 }}>{d?.model.kind ?? "—"}</span>}
          sub={<span className="mono" style={{ fontSize: 11 }}>{d?.model.version ?? "loading"}</span>} />
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Transaction volume" sub="Transactions scored per 10-second interval (up to the last 10 minutes)">
          {d && <TimeSeriesChart data={activity} series={[{ key: "transactions", label: "Transactions", color: "var(--s1)", kind: "area" }]}
            xFormat={fmtTime} label="Transactions per 10 seconds" />}
        </Card>
        <Card title="Alerts" sub="Review + decline recommendations per 10-second interval">
          {d && <TimeSeriesChart data={activity} series={[{ key: "alerts", label: "Alerts", color: "var(--s2)", kind: "bar" }]}
            xFormat={fmtTime} label="Alerts per 10 seconds" />}
        </Card>
      </div>

      <div className="grid cols-2-1" style={{ marginTop: 16 }}>
        <Card title="Risk distribution" sub="Calibrated fraud probability of every scored transaction (log-spaced bins)">
          {s && <BarChart data={s.probability_histogram.map((b) => ({ label: b.bin.split("–")[0], value: b.count }))}
            valueFormat={(v) => fmtCompact(v)} label="Histogram of fraud probability" xLabel="Fraud probability (bin lower edge)" />}
          {s && (
            <div className="row" style={{ marginTop: 10 }}>
              {(["low", "medium", "high"] as const).map((k) => (
                <span key={k} className="row small ink2" style={{ gap: 6 }}><RiskBadge level={k} />{fmtInt(s.risk_distribution[k] ?? 0)}</span>
              ))}
            </div>
          )}
        </Card>
        <Card title="Latest alerts" sub="Newest review/decline recommendations" actions={<a href="#/cases" className="small">Open queue →</a>} flush>
          {alerts.data?.items.length ? (
            <table className="data">
              <tbody>
                {alerts.data.items.map((t) => (
                  <tr key={t.transaction_id} className="clickable" onClick={() => (window.location.hash = `#/cases/${t.transaction_id}`)}>
                    <td><div className="mono" style={{ fontSize: 11.5 }}>{t.transaction_id}</div>
                      <div className="small muted">{fmtTime(t.received_at)} {t.synthetic && <SyntheticTag />}</div></td>
                    <td className="num">{fmtMoney(t.amount)}<div className="small muted">{fmtProb(t.fraud_probability)}</div></td>
                    <td><ActionBadge action={t.action} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div className="empty">No alerts yet. Start a replay to stream held-out transactions.</div>}
        </Card>
      </div>
      <p className="note">{d?.exposure_note} Flagged amount: {fmtMoney(s?.flagged_amount, true)} of {fmtMoney(s?.volume, true)} processed.</p>
    </>
  );
}
