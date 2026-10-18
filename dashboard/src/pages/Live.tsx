import { useEffect, useMemo, useRef, useState } from "react";
import { api, type TxnSummary } from "../lib/api";
import { fmtMoney, fmtEventTime, fmtInt, fmtMs, fmtTime } from "../lib/format";
import { usePoll, useTransactionStream } from "../lib/hooks";
import { ActionBadge, Card, PageHead, ProbBar, ReplayBar, RiskBadge, StatusBadge, SyntheticTag } from "../components/ui";

type Filter = "all" | "alerts" | "review" | "decline";

export default function Live() {
  const [paused, setPaused] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [risk, setRisk] = useState("");
  const [search, setSearch] = useState("");
  const stream = useTransactionStream(300, paused);
  const replay = usePoll(api.replayStatus, 1500);
  const initial = usePoll(() => api.transactions({ limit: 100 }), 0);
  const seen = useRef<Set<string>>(new Set());
  const [fresh, setFresh] = useState<Set<string>>(new Set());

  const rows = useMemo(() => {
    const byId = new Map<string, TxnSummary>();
    for (const r of [...stream.events, ...(initial.data?.items ?? [])]) if (!byId.has(r.transaction_id)) byId.set(r.transaction_id, r);
    return [...byId.values()].sort((a, b) => b.received_at - a.received_at);
  }, [stream.events, initial.data]);

  useEffect(() => {
    const nu = new Set<string>();
    for (const r of stream.events.slice(0, 40)) if (!seen.current.has(r.transaction_id)) { nu.add(r.transaction_id); seen.current.add(r.transaction_id); }
    if (nu.size) setFresh(nu);
  }, [stream.events]);

  const visible = rows.filter((r) =>
    (filter === "all" || (filter === "alerts" ? r.action !== "approve" : r.action === filter)) &&
    (!risk || r.risk_level === risk) &&
    (!search || r.transaction_id.toLowerCase().includes(search.toLowerCase())),
  ).slice(0, 150);

  return (
    <>
      <PageHead title="Live transaction monitor"
        desc="Every transaction scored by the API or the replay engine, streamed over server-sent events." />
      <ReplayBar status={replay.data} onChange={replay.refresh} />
      <Card
        title={<span className="row" style={{ gap: 8 }}><span className={`live-dot${stream.connected && !paused ? "" : " off"}`} />
          {paused ? "Feed paused" : stream.connected ? "Streaming" : "Connecting…"}</span>}
        sub={`${fmtInt(stream.received)} events received this session${stream.dropped ? ` · ${stream.dropped} dropped by slow-consumer protection` : ""}`}
        actions={
          <div className="row">
            <input className="input" placeholder="Search transaction ID" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search transaction ID" style={{ width: 200 }} />
            <div className="seg" role="group" aria-label="Filter by action">
              {(["all", "alerts", "review", "decline"] as Filter[]).map((f) => (
                <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>{f[0].toUpperCase() + f.slice(1)}</button>
              ))}
            </div>
            <select className="select" value={risk} onChange={(e) => setRisk(e.target.value)} aria-label="Risk level">
              <option value="">All risk levels</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option>
            </select>
            <button className="btn" onClick={() => setPaused((p) => !p)}>{paused ? "Resume feed" : "Pause feed"}</button>
          </div>
        }
        flush
      >
        <div className="table-wrap scroll-y" style={{ maxHeight: "calc(100vh - 330px)", minHeight: 300 }}>
          <table className="data">
            <thead>
              <tr>
                <th>Received</th><th>Transaction</th><th>Event time</th><th className="num">Amount</th><th>Risk score</th>
                <th>Risk</th><th>Recommended</th><th>Case</th><th className="num">Inference</th><th />
              </tr>
            </thead>
            <tbody>
              {visible.map((t) => (
                <tr key={t.transaction_id} className={`clickable${fresh.has(t.transaction_id) ? " fresh" : ""}`}
                  onClick={() => (window.location.hash = `#/cases/${t.transaction_id}`)}>
                  <td className="nowrap">{fmtTime(t.received_at)}</td>
                  <td><span className="mono">{t.transaction_id}</span> {t.synthetic && <SyntheticTag />}
                    {t.top_reason && t.action !== "approve" && <div className="small ink2 truncate" title={t.top_reason}>{t.top_reason}</div>}</td>
                  <td className="nowrap small ink2">{fmtEventTime(t.event_time)}</td>
                  <td className="num">{fmtMoney(t.amount)}</td>
                  <td><ProbBar p={t.fraud_probability} level={t.risk_level} /></td>
                  <td><RiskBadge level={t.risk_level} /></td>
                  <td><ActionBadge action={t.action} /></td>
                  <td><StatusBadge status={t.review_status} /></td>
                  <td className="num small ink2">{fmtMs(t.latency_ms)}</td>
                  <td><a href={`#/cases/${t.transaction_id}`} onClick={(e) => e.stopPropagation()} className="small">
                    {t.action === "approve" ? "Explain" : "Investigate"}</a></td>
                </tr>
              ))}
            </tbody>
          </table>
          {!visible.length && <div className="empty">No transactions match. Start a replay above or POST to /api/v1/transactions/score.</div>}
        </div>
      </Card>
    </>
  );
}
