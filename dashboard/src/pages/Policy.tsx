import { useEffect, useState } from "react";
import { api, type PolicyInput, type Simulation } from "../lib/api";
import { fmtEur, fmtInt, fmtPct } from "../lib/format";
import { useDebounced, usePoll } from "../lib/hooks";
import { usePublicConfig } from "../lib/config";
import { CurveChart } from "../components/charts";
import { Card, ErrorNote, Icon, PageHead, Stat } from "../components/ui";

// Review threshold slider works on a log scale (0.01% .. 90%).
const toSlider = (p: number) => Math.round(((Math.log10(p) + 4) / (Math.log10(0.9) + 4)) * 1000);
const fromSlider = (v: number) => 10 ** ((v / 1000) * (Math.log10(0.9) + 4) - 4);

export default function Policy() {
  const pol = usePoll(api.policy, 0);
  const [draft, setDraft] = useState<PolicyInput | null>(null);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const debounced = useDebounced(draft, 250);
  const cfg = usePublicConfig();

  useEffect(() => {
    if (pol.data && !draft) {
      const c = pol.data.config;
      setDraft({ review_threshold: c.review_threshold, decline_threshold: c.decline_threshold, expected_loss_review: c.expected_loss_review,
        max_reviews_per_hour: c.max_reviews_per_hour, review_cost: c.cost_assumptions.review_cost, false_decline_cost: c.cost_assumptions.false_decline_cost });
    }
  }, [pol.data, draft]);

  useEffect(() => {
    if (!debounced) return;
    if (debounced.decline_threshold <= debounced.review_threshold) { setError("Decline threshold must exceed the review threshold."); return; }
    api.simulate(debounced).then((s) => { setSim(s); setError(null); }).catch((e) => setError(e.message));
  }, [debounced]);

  if (!pol.data || !draft) return <><PageHead title="Decision policy" /><ErrorNote error={pol.error} /></>;
  const set = (patch: Partial<PolicyInput>) => { setDraft({ ...draft, ...patch }); setSaved(null); };
  const apply = async () => {
    try { const r = await api.updatePolicy(draft); setSaved(r.version); pol.refresh(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const resetToModel = () => pol.data?.model_review_threshold && set({ review_threshold: pol.data.model_review_threshold });
  const res = sim?.result;

  return (
    <>
      <PageHead title="Decision policy"
        desc="The model estimates how likely a payment is to be fraud; this policy decides what to do about it. Changes apply to new decisions and are versioned." />
      <div className="callout warn" style={{ marginBottom: 16 }}><Icon.info /><span>{pol.data.disclaimer} The what-if below runs on the labelled
        held-out test window. Tuning thresholds here would be tuning on test data; the default review threshold was chosen on validation.</span></div>

      <div className="grid cols-1-2">
        <Card title="Policy settings" sub={<>Active: <span className="mono">{pol.data.version}</span></>}>
          <div className="stack" style={{ gap: 14 }}>
            <label className="field">
              <span className="spread"><span>Review threshold</span><b className="num">{fmtPct(draft.review_threshold, 2)}</b></span>
              <input type="range" min={0} max={1000} value={toSlider(draft.review_threshold)} onChange={(e) => set({ review_threshold: fromSlider(+e.target.value) })} />
              <span className="small muted">Model default (validation cost-optimal): {fmtPct(pol.data.model_review_threshold, 2)}
                {" "}<button className="btn sm" onClick={resetToModel} style={{ marginLeft: 6 }}>Use default</button></span>
            </label>
            <label className="field">
              <span className="spread"><span>Recommend decline at</span><b className="num">{fmtPct(draft.decline_threshold, 1)}</b></span>
              <input type="range" min={500} max={999} value={Math.round(draft.decline_threshold * 1000)} onChange={(e) => set({ decline_threshold: +e.target.value / 1000 })} />
            </label>
            <div className="grid cols-2" style={{ gap: 10 }}>
              <label className="field">Review cost (€)
                <input className="input" type="number" min={0} step={0.5} value={draft.review_cost} onChange={(e) => set({ review_cost: +e.target.value })} /></label>
              <label className="field">False-decline cost (€)
                <input className="input" type="number" min={0} step={1} value={draft.false_decline_cost} onChange={(e) => set({ false_decline_cost: +e.target.value })} /></label>
              <label className="field">Review capacity (per hour)
                <input className="input" type="number" min={1} step={5} value={draft.max_reviews_per_hour} onChange={(e) => set({ max_reviews_per_hour: Math.max(1, +e.target.value) })} /></label>
              <label className="field" style={{ justifyContent: "flex-end" }}>
                <span className="row" style={{ gap: 6 }}><input type="checkbox" checked={draft.expected_loss_review} onChange={(e) => set({ expected_loss_review: e.target.checked })} />
                  Also review when p × amount ≥ review cost</span></label>
            </div>
            <div className="row">
              <button className="btn primary" onClick={apply} disabled={!cfg.policy_changes_allowed}
                title={cfg.policy_changes_allowed ? undefined : "Disabled in the public demo; the what-if still works"}>Apply to live decisions</button>
              {!cfg.policy_changes_allowed && <span className="small muted">Disabled in the public demo; the what-if still works.</span>}
              {saved && <span className="small ink2">Applied as <span className="mono">{saved}</span></span>}
            </div>
            <ErrorNote error={error} />
          </div>
        </Card>

        <div className="stack">
          <div className="kpis">
            <Stat label="Frauds stopped" value={res ? `${res.frauds_stopped} / ${res.frauds}` : "—"} sub={res ? `${fmtPct(res.frauds_stopped / Math.max(1, res.frauds))} of frauds` : undefined} />
            <Stat label="Fraud value stopped" value={res ? fmtEur(res.fraud_amount_stopped, true) : "—"} sub={res ? `of ${fmtEur(res.fraud_amount_total, true)}` : undefined} />
            <Stat label="Manual reviews" value={res ? fmtInt(res.reviewed) : "—"} sub={sim ? `${(res!.reviewed / sim.window_hours).toFixed(1)} per hour` : undefined} />
            <Stat label="Declines" value={res ? fmtInt(res.declined) : "—"} sub={res ? `${res.legit_declined} legitimate customers declined` : undefined} />
            <Stat label="Expected total cost" value={res ? fmtEur(res.cost.total, true) : "—"}
              sub={res ? `loss ${fmtEur(res.cost.fraud_loss, true)} · review ${fmtEur(res.cost.review, true)} · friction ${fmtEur(res.cost.false_declines, true)}` : undefined} />
          </div>
          <Card title="Investigation capacity" sub={sim ? `If analysts review only the top-scored alerts (held-out window of ${sim.window_hours.toFixed(1)} h)` : ""}>
            {sim && (
              <>
                <CurveChart label="Recall vs review capacity" xLabel="Reviews per hour" yLabel="Share of frauds caught"
                  xDomain={[0, Math.max(...sim.capacity.map((c) => c.reviews_per_hour))]} xFormat={(v) => `${Math.round(v)}`} yFormat={(v) => fmtPct(v, 0)}
                  curves={[{ key: "recall", label: "Recall", color: "var(--s1)", points: sim.capacity.map((c) => [c.reviews_per_hour, c.recall]) },
                    { key: "precision", label: "Precision of reviewed alerts", color: "var(--s2)", points: sim.capacity.map((c) => [c.reviews_per_hour, c.precision]) }]} height={220} />
                <table className="data" style={{ marginTop: 8 }}>
                  <thead><tr><th className="num">Reviews / hour</th><th className="num">Reviews</th><th className="num">Frauds caught</th><th className="num">Precision</th><th className="num">Residual loss</th></tr></thead>
                  <tbody>{sim.capacity.map((c) => (
                    <tr key={c.reviews_per_hour}><td className="num">{c.reviews_per_hour}</td><td className="num">{fmtInt(c.reviews)}</td>
                      <td className="num">{c.frauds_caught} ({fmtPct(c.recall, 0)})</td><td className="num">{fmtPct(c.precision)}</td>
                      <td className="num">{fmtEur(c.residual_fraud_loss, true)}</td></tr>
                  ))}</tbody>
                </table>
              </>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
