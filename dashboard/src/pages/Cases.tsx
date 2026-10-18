import { useEffect, useState } from "react";
import { api, type Explanation, type Linked, type Signal, type TxnDetail } from "../lib/api";
import { usePublicConfig } from "../lib/config";
import { STATUS_LABEL, fmtAgo, fmtDateTime, fmtMoney, fmtEventTime, fmtMs, fmtNum, fmtPct, fmtProb, fmtInt } from "../lib/format";
import { usePoll } from "../lib/hooks";
import { HBarChart } from "../components/charts";
import { ActionBadge, Card, ErrorNote, Icon, PageHead, RiskBadge, StatusBadge, SyntheticTag } from "../components/ui";

const QUEUE_FILTERS = [
  { key: "open", label: "Open" }, { key: "escalated", label: "Escalated" },
  { key: "confirmed_fraud", label: "Fraud" }, { key: "legitimate", label: "Legit" }, { key: "", label: "All alerts" },
];

export default function Cases({ id }: { id?: string }) {
  const [status, setStatus] = useState("open");
  const queue = usePoll(() => api.transactions({ alerts_only: true, review_status: status || undefined, limit: 100 }), 4000, [status]);

  useEffect(() => {
    if (!id && queue.data?.items.length) window.location.replace(`#/cases/${queue.data.items[0].transaction_id}`);
  }, [id, queue.data]);

  return (
    <>
      <PageHead title="Fraud investigation"
        desc="Review alerts, inspect the model's reasoning and record a decision. Decisions are stored with an audit trail." />
      <div className="case-layout">
        <Card title="Alert queue" sub={`${queue.data?.total ?? 0} ${STATUS_LABEL[status]?.toLowerCase() ?? "total"} alerts`} flush
          actions={<select className="select" value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Queue filter">
            {QUEUE_FILTERS.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
          </select>}>
          <div className="queue">
            {queue.data?.items.map((t) => (
              <div key={t.transaction_id} className={`queue-item${t.transaction_id === id ? " active" : ""}`}
                onClick={() => (window.location.hash = `#/cases/${t.transaction_id}`)} role="button" tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && (window.location.hash = `#/cases/${t.transaction_id}`)}>
                <span className="id">{t.transaction_id}</span>
                <span className="num" style={{ fontWeight: 600 }}>{fmtProb(t.fraud_probability)}</span>
                <span className="small muted">{fmtMoney(t.amount)} · {fmtAgo(t.received_at)}</span>
                {t.top_reasons?.[0] && <span className="small ink2 truncate" style={{ gridColumn: "1 / -1" }} title={t.top_reasons[0]}>{t.top_reasons[0]}</span>}
                <span><ActionBadge action={t.action} /></span>
              </div>
            ))}
            {!queue.data?.items.length && <div className="empty">Queue is empty for this filter.</div>}
          </div>
        </Card>
        {id ? <Workspace id={id} onDecision={queue.refresh} /> : <Card><div className="empty">Select an alert.</div></Card>}
      </div>
    </>
  );
}

function Workspace({ id, onDecision }: { id: string; onDecision: () => void }) {
  const [txn, setTxn] = useState<TxnDetail | null>(null);
  const [exp, setExp] = useState<{ explanation: Explanation; signals: Record<string, Signal> } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [showFeatures, setShowFeatures] = useState(false);
  const cfg = usePublicConfig();

  const load = async () => {
    try {
      setError(null);
      const t = await api.transaction(id);
      setTxn(t);
      setExp(await api.explanation(id));
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  useEffect(() => { setTxn(null); setExp(null); setNotes(""); load(); }, [id]); // eslint-disable-line

  const decide = async (decision: string) => {
    setSaving(true);
    try { await api.review(id, decision, notes); setNotes(""); await load(); onDecision(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setSaving(false); }
  };

  if (error && !txn) return <Card><ErrorNote error={error} /></Card>;
  if (!txn) return <Card><div className="empty">Loading…</div></Card>;
  const factors = exp?.explanation.top_factors ?? [];
  const risky = factors.filter((f) => f.contribution > 0 && f.reason).slice(0, 4);
  const shown = (f: (typeof factors)[number]) => f.display_value ?? fmtNum(f.value, 3);
  const closed = txn.review_status === "confirmed_fraud" || txn.review_status === "legitimate";

  return (
    <div className="stack">
      <Card>
        <div className="spread" style={{ alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <div className="row"><span className="mono" style={{ fontSize: 13 }}>{txn.transaction_id}</span>
              {txn.synthetic && <SyntheticTag />}<StatusBadge status={txn.review_status} /></div>
            <div className="row" style={{ marginTop: 12, alignItems: "flex-end", gap: 16 }}>
              <div><div className="small muted">Fraud probability</div><div className="hero-prob">{fmtProb(txn.fraud_probability)}</div></div>
              <div><div className="small muted">Amount</div><div style={{ fontSize: 22, fontWeight: 620 }}>{fmtMoney(txn.amount)}</div></div>
              <div className="row"><RiskBadge level={txn.risk_level} /><ActionBadge action={txn.action} /></div>
            </div>
          </div>
          <dl className="kv" style={{ minWidth: 240 }}>
            <dt>Scored</dt><dd>{fmtDateTime(txn.received_at)}</dd>
            <dt>Event time</dt><dd>{fmtEventTime(txn.event_time)}</dd>
            <dt>Source</dt><dd>{txn.source}{txn.scenario ? ` · ${txn.scenario}` : ""}</dd>
            <dt>Model</dt><dd className="mono" style={{ fontSize: 11.5 }}>{txn.model_version}</dd>
            <dt>Policy</dt><dd className="mono" style={{ fontSize: 11.5 }}>{txn.policy_version}</dd>
          </dl>
        </div>
      </Card>

      <div className="grid cols-2">
        <Card title="Why this recommendation" sub="Decision policy applied to the calibrated probability">
          <ul style={{ margin: 0, paddingLeft: 18 }}>{txn.reasons?.map((r) => <li key={r}>{r}</li>)}</ul>
          <dl className="kv" style={{ marginTop: 12 }}>
            <dt>Expected loss</dt><dd>{fmtMoney(txn.expected_loss)} <span className="muted small">(probability × amount)</span></dd>
            <dt>Raw model score</dt><dd>{fmtNum(txn.raw_score, 4)} <span className="muted small">(before calibration)</span></dd>
            <dt>Inference time</dt><dd>{fmtMs(txn.latency_ms)}</dd>
          </dl>
          {txn.over_capacity && <p className="callout warn" style={{ marginTop: 12 }}>Routed while the review team was over its hourly capacity.</p>}
          {txn.quality.length > 0 && (
            <div className="callout warn" style={{ marginTop: 12, flexDirection: "column" }}>
              <strong>Data-quality warnings</strong>{txn.quality.map((q) => <span key={q}>{q}</span>)}
            </div>
          )}
        </Card>
        <Card title="Model signals" sub="Unsupervised detectors are context for analysts unless marked as scoring inputs">
          <div className="signal">
            <span>Calibrated fraud probability <span className="muted small">· model + Platt calibration</span></span><b>{fmtPct(txn.fraud_probability, 2)}</b>
          </div>
          {exp && Object.entries(exp.signals).map(([k, s]) => (
            <div className="signal" key={k}>
              <span>{k === "iforest_score" ? "Isolation Forest anomaly" : "Autoencoder reconstruction error"}
                <span className="muted small"> · {s.used_in_score ? "used in score" : "context only"}</span></span>
              <b>{s.percentile_vs_normal == null ? fmtNum(s.value) : `${s.percentile_vs_normal.toFixed(1)}th pct`}</b>
              <div className="meter" title="Percentile vs. legitimate validation transactions"><div style={{ width: `${s.percentile_vs_normal ?? 0}%` }} /></div>
            </div>
          ))}
          <p className="note">Percentiles compare this transaction with legitimate transactions from the validation period.</p>
        </Card>
      </div>

      <Card title="Model explanation" sub={exp ? `${exp.explanation.method} · contributions in ${exp.explanation.space}` : "Loading…"}>
        {exp && (
          <>
            {risky.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div className="small muted" style={{ marginBottom: 4 }}>Top risk factors, in plain language</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>{risky.map((f) => <li key={f.feature}>{f.reason}</li>)}</ul>
              </div>
            )}
            <div className="legend"><span><i className="key box" style={{ background: "var(--div-pos)" }} />Pushes toward fraud</span>
              <span><i className="key box" style={{ background: "var(--div-neg)" }} />Pushes toward legitimate</span></div>
            <HBarChart diverging label="Feature contributions" labelWidth={110}
              rows={factors.map((f) => ({
                label: f.feature, value: f.contribution,
                detail: `value ${shown(f)}${f.training_percentile != null ? ` · ${f.training_percentile.toFixed(1)}th percentile in training` : ""}`,
              }))}
              valueFormat={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`} />
            <div className="table-wrap" style={{ marginTop: 10 }}>
              <table className="data">
                <thead><tr><th>Feature</th><th className="num">Value</th><th className="num">Training percentile</th><th className="num">Contribution</th><th>Meaning</th></tr></thead>
                <tbody>{factors.map((f) => (
                  <tr key={f.feature}>
                    <td className="mono">{f.feature}</td><td className="num">{shown(f)}</td>
                    <td className="num">{f.training_percentile == null ? "—" : f.training_percentile.toFixed(1)}</td>
                    <td className="num">{f.contribution > 0 ? "+" : ""}{f.contribution.toFixed(3)}</td>
                    <td className="small ink2">{f.description}</td>
                  </tr>))}
                </tbody>
              </table>
            </div>
            <p className="callout" style={{ marginTop: 12 }}><Icon.info /> <span>{exp.explanation.disclaimer} Base value {exp.explanation.base_value.toFixed(2)};
              other features {exp.explanation.other_features_contribution >= 0 ? "+" : ""}{exp.explanation.other_features_contribution.toFixed(2)}.
              {cfg.profile === "ieee"
                ? " Velocity features (v_*) are computed online at scoring time from earlier transactions only."
                : " V1–V28 are anonymised PCA components, so their real-world meaning is not available."}</span></p>
          </>
        )}
      </Card>

      {cfg.linked_activity && <LinkedActivity id={id} />}

      <div className="grid cols-2">
        <Card title="Analyst decision" sub={closed ? "Case closed — you can still add notes or reopen by escalating" : "Record the outcome of your review"}>
          <textarea className="input" rows={3} maxLength={2000} placeholder="Investigation notes (optional)" value={notes}
            onChange={(e) => setNotes(e.target.value)} aria-label="Investigation notes" />
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn danger" disabled={saving} onClick={() => decide("confirmed_fraud")}>Confirm fraud</button>
            <button className="btn good" disabled={saving} onClick={() => decide("legitimate")}>Mark legitimate</button>
            <button className="btn" disabled={saving} onClick={() => decide("escalated")}>Escalate</button>
            <button className="btn" disabled={saving || !notes.trim()} onClick={() => decide("note")}>Add note</button>
          </div>
          {closed && txn.ground_truth_revealed != null && (
            <p className="callout" style={{ marginTop: 12 }}>
              Historical label for this replayed transaction: <b style={{ marginLeft: 4 }}>{txn.ground_truth_revealed ? "fraud" : "legitimate"}</b>.
              Shown only after a decision, to simulate labels that arrive later (e.g. chargebacks).
            </p>
          )}
          <ErrorNote error={error} />
        </Card>
        <Card title="Alert history" sub="Audit trail for this transaction">
          <ul className="timeline">
            <li><b>Scored</b> <span className="muted small">{fmtDateTime(txn.received_at)}</span><div className="small ink2">Recommended {txn.action} at {fmtProb(txn.fraud_probability)}</div></li>
            {txn.history.map((h) => (
              <li key={h.id}><b>{h.event === "note" ? "Note" : STATUS_LABEL[h.decision ?? ""] ?? h.decision}</b>{" "}
                <span className="muted small">{fmtDateTime(h.at)} · {h.actor}</span>
                {h.notes && <div className="small ink2">{h.notes}</div>}</li>
            ))}
          </ul>
        </Card>
      </div>

      <Card title="Transaction features" sub="Raw inputs as received" actions={<button className="btn sm" onClick={() => setShowFeatures((s) => !s)}>{showFeatures ? "Hide" : "Show all"}</button>}>
        {showFeatures ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "6px 14px" }}>
            {Object.entries(txn.features).map(([k, v]) => (
              <div key={k} className="spread small"><span className="muted mono">{k}</span>
                <span className="num">{typeof v === "number" ? fmtNum(v, 3) : v ?? "—"}</span></div>
            ))}
          </div>
        ) : <p className="muted small" style={{ margin: 0 }}>{cfg.profile === "ieee"
          ? `${Object.keys(txn.features).length} values: the authorisation-time field contract plus the velocity features as computed at scoring time.`
          : "30 raw inputs (Time, Amount, V1–V28)."}</p>}
      </Card>
    </div>
  );
}

function LinkedActivity({ id }: { id: string }) {
  const [data, setData] = useState<Linked | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setData(null);
    api.linked(id).then(setData).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [id]);
  return (
    <Card title="Linked activity" sub="Transactions in this session that share a card, card + e-mail domain, card + region or device model">
      <ErrorNote error={error} />
      {!data && !error && <div className="empty">Loading…</div>}
      {data && !data.links.length && <p className="muted small" style={{ margin: 0 }}>No entity keys on this transaction (missing card or device fields).</p>}
      {data?.links.map((l) => (
        <div key={l.type} style={{ marginBottom: 14 }}>
          <div className="spread" style={{ marginBottom: 6 }}>
            <span className="row" style={{ gap: 8 }}><b>{l.label}</b>
              <span className={`badge ${l.strength === "strong" ? "info" : ""}`}><span className="dot" />{l.strength} link</span></span>
            <span className="small ink2">{fmtInt(l.earlier)} earlier · {fmtInt(l.earlier_24h)} in the 24 h before · {fmtInt(l.total)} in total</span>
          </div>
          {l.transactions.length > 0 && (
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>Transaction</th><th>Event time</th><th className="num">Amount</th><th className="num">Probability</th><th>Recommended</th><th>Case</th></tr></thead>
                <tbody>{l.transactions.map((t) => (
                  <tr key={t.transaction_id} className="clickable" onClick={() => (window.location.hash = `#/cases/${t.transaction_id}`)}>
                    <td className="mono">{t.transaction_id}</td><td className="nowrap small ink2">{fmtEventTime(t.event_time)}</td>
                    <td className="num">{fmtMoney(t.amount)}</td><td className="num">{fmtProb(t.fraud_probability)}</td>
                    <td><ActionBadge action={t.action} /></td><td><StatusBadge status={t.review_status} /></td>
                  </tr>))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
      {data?.note && <p className="note">{data.note}</p>}
    </Card>
  );
}
