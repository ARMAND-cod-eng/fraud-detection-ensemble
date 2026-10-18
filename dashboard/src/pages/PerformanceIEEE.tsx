import type { IEEEEvaluation, Versions } from "../lib/api";
import { fmtInt, fmtMs, fmtPct } from "../lib/format";
import { ConfusionMatrix, CurveChart, HBarChart } from "../components/charts";
import { Card, Icon, PageHead } from "../components/ui";

// Fixed categorical slots: colour follows the variant, never its rank.
const VARIANTS = [
  { key: "raw_contract", label: "Raw contract fields", color: "var(--s2)" },
  { key: "raw_plus_velocity", label: "Raw + velocity features", color: "var(--s1)" },
];
const label = (k: string) => VARIANTS.find((v) => v.key === k)?.label ?? k;

/** FraudShield v2: IEEE-CIS on authorisation-time fields + online velocity features. */
export default function PerformanceIEEE({ e, versions }: { e: IEEEEvaluation; versions: Versions | null }) {
  const sel = e.selected_model;
  const ci = e.test_bootstrap_95ci.pr_auc;
  const delta = ci.raw_plus_velocity;
  const op = e.selected_operating_points;
  const r = op.review_threshold_test;
  const lat = e.latency_online;
  const sp = e.dataset.split;
  // The gate report lives in the registry history (the artifact is written before promotion).
  const promotion = (versions?.history ?? []).filter((h) => h.version === e.version && h.gate).at(-1);
  const gate = promotion?.gate ?? (e.promotion ? { passed: e.promotion.promoted, checks: e.promotion.report.checks } : null);
  const failed = gate ? gate.checks.filter((c) => !c.passed) : [];
  const servedAnyway = !!gate && !gate.passed && versions?.serving === e.version;
  const day = (s: number) => (s / 86400).toFixed(0);
  const bt = e.backtest.blocks;
  const btMax = Math.max(...bt.flatMap((b) => [b.static_pr_auc, b.retrained_pr_auc]));

  return (
    <>
      <PageHead title="Model performance · FraudShield v2"
        desc={<>IEEE-CIS model <span className="mono">{e.version}</span>, created {new Date(e.created_at).toLocaleString()}.
          Only fields available when a card is authorised ({e.protocol.field_contract.length} raw fields) plus {e.protocol.velocity_features.length} velocity
          features from the online feature store. Out-of-time split; the velocity features had to earn their place on validation; test scored once.</>} />

      {gate && !gate.passed && (
        <div className="callout warn" style={{ marginBottom: 16 }} role="note">
          <Icon.info /><span><b>The pre-registered promotion gate rejected this model</b> ({failed.map((c) => {
            const d = c.detail as { value?: number; limit?: number; op?: string };
            return d && d.value != null ? `${c.name} ${d.value.toFixed(3)} ${d.op === ">=" ? "<" : ">"} ${d.limit}` : c.name;
          }).join("; ")}). The limits were set before any v2 result was seen and were not lowered afterwards.
            {servedAnyway ? " It is served here through an explicit, logged override for demonstration only." : ""}</span>
        </div>
      )}

      <div className="kpis">
        {[
          ["Selected variant", label(sel), "pre-registered rule on validation"],
          ["Test PR-AUC", e.test[sel].pr_auc.toFixed(3), `95% CI ${ci[sel].ci_low.toFixed(3)}–${ci[sel].ci_high.toFixed(3)}`],
          ["Velocity gain (test)", `${delta.delta_vs_raw_contract! >= 0 ? "+" : ""}${delta.delta_vs_raw_contract!.toFixed(3)}`,
            `paired 95% CI ${delta.delta_ci_low!.toFixed(3)} to ${delta.delta_ci_high!.toFixed(3)}`],
          ["All-431-column ceiling", e.reference_all_431_columns_test_pr_auc?.toFixed(3) ?? "—",
            e.share_of_gap_to_vendor_features_recovered == null ? "v1 benchmark" : `velocity recovers ${fmtPct(e.share_of_gap_to_vendor_features_recovered, 0)} of the gap`],
          ["Test ROC-AUC", e.test[sel].roc_auc.toFixed(3), `raw contract ${e.test.raw_contract.roc_auc.toFixed(3)}`],
          ["Online latency (p95)", fmtMs(lat.single_row_p95_ms), `feature store ${fmtMs(lat.feature_store_p95_ms)} · p50 ${fmtMs(lat.single_row_p50_ms)}`],
        ].map(([l, v, s]) => (
          <div className="card stat" key={l}><div className="label">{l}</div><div className="value" style={{ fontSize: 21 }}>{v}</div><div className="sub">{s}</div></div>
        ))}
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Precision–recall (test)" sub="Fraud is ~3.4% of IEEE-CIS traffic">
          <CurveChart label="Precision-recall curves" xLabel="Recall" yLabel="Precision"
            curves={VARIANTS.filter((v) => e.curves_test[v.key]).map((v) => ({ key: v.key, label: `${v.label} · AP ${e.test[v.key].pr_auc.toFixed(3)}`, color: v.color, points: e.curves_test[v.key].pr }))} />
        </Card>
        <Card title="Monthly backtest" sub="30-day blocks; the month before each scored month is excluded (labels not yet mature)">
          <CurveChart label="Backtest PR-AUC by month" xLabel="Scored month (30-day block)" yLabel="PR-AUC"
            xDomain={[Math.min(...bt.map((b) => b.test_block)), Math.max(...bt.map((b) => b.test_block))]} yDomain={[0, Math.ceil(btMax * 12) / 10]}
            xFormat={(v) => (Number.isInteger(v) ? `M${v}` : "")} yFormat={(v) => v.toFixed(2)}
            curves={[
              { key: "static", label: "Static (trained on months 0–1 once)", color: "var(--s2)", points: bt.map((b) => [b.test_block, b.static_pr_auc]) },
              { key: "retrained", label: "Retrained monthly", color: "var(--s1)", points: bt.map((b) => [b.test_block, b.retrained_pr_auc]) },
            ]} />
          <table className="data" style={{ marginTop: 8 }}>
            <thead><tr><th>Month</th><th className="num">Frauds</th><th className="num">Static</th><th className="num">Retrained</th><th className="num">Δ</th></tr></thead>
            <tbody>{bt.map((b) => (
              <tr key={b.test_block}><td>M{b.test_block} <span className="muted small">(train {b.train_blocks_retrained})</span></td><td className="num">{fmtInt(b.frauds)}</td>
                <td className="num">{b.static_pr_auc.toFixed(3)}</td><td className="num">{b.retrained_pr_auc.toFixed(3)}</td>
                <td className="num">{b.retrained_pr_auc - b.static_pr_auc >= 0 ? "+" : ""}{(b.retrained_pr_auc - b.static_pr_auc).toFixed(3)}</td></tr>
            ))}</tbody>
          </table>
          <p className="note">The static model decays month over month; retraining with mature labels holds performance. In the first scored month both models use the same data.</p>
        </Card>
      </div>

      <Card title="Variants vs. the vendor-feature ceiling" sub="Validation drove selection; test with 95% bootstrap CIs and paired differences vs. the raw contract" flush className="mt">
        <div className="table-wrap">
          <table className="data">
            <thead><tr>
              <th>Variant</th><th className="num">Inputs</th><th className="num">Val PR-AUC</th><th className="num">Test PR-AUC [95% CI]</th><th className="num">Δ vs raw [95% CI]</th>
              <th className="num">ROC-AUC</th><th className="num">Recall @ 1% FPR</th><th className="num">Prec. @ top 0.5%</th>
            </tr></thead>
            <tbody>
              {VARIANTS.filter((v) => e.test[v.key]).map((v) => {
                const c = ci[v.key];
                const t = e.test[v.key];
                const top = Object.keys(t).find((k) => k.startsWith("precision@top0.50%")) ?? "";
                return (
                  <tr key={v.key} style={v.key === sel ? { background: "var(--accent-wash)" } : undefined}>
                    <td className="nowrap"><b>{v.label}</b>{v.key === sel && <span className="badge info" style={{ marginLeft: 8 }}><span className="dot" />selected</span>}</td>
                    <td className="num">{String(e.candidates[v.key]?.info.inputs ?? "—")}</td>
                    <td className="num">{e.validation[v.key].pr_auc.toFixed(3)}</td>
                    <td className="num nowrap">{t.pr_auc.toFixed(3)} <span className="muted">[{c.ci_low.toFixed(3)}, {c.ci_high.toFixed(3)}]</span></td>
                    <td className="num nowrap">{c.delta_vs_raw_contract == null ? "—" : <>{c.delta_vs_raw_contract >= 0 ? "+" : ""}{c.delta_vs_raw_contract.toFixed(3)} <span className="muted">[{c.delta_ci_low!.toFixed(3)}, {c.delta_ci_high!.toFixed(3)}]</span></>}</td>
                    <td className="num">{t.roc_auc.toFixed(3)}</td>
                    <td className="num">{fmtPct(t["recall@fpr=0.01"])}</td>
                    <td className="num">{fmtPct(t[top])}</td>
                  </tr>
                );
              })}
              {e.reference_all_431_columns_test_pr_auc != null && (
                <tr className="muted"><td>All 431 columns<div className="small">incl. vendor C/D/V/id_ aggregates (v1 benchmark)</div></td>
                  <td className="num">431</td><td className="num">—</td><td className="num">{e.reference_all_431_columns_test_pr_auc.toFixed(3)}</td><td className="num">—</td><td className="num">—</td><td className="num">—</td><td className="num">—</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div style={{ padding: "4px 16px 14px" }}>
          <p className="note"><b>Selection log:</b> {e.protocol.selection_log.join(" · ")}</p>
          <p className="note">The vendor aggregates encode history a single merchant's stream does not contain, so the ceiling is not reachable from the field contract;
            the honest comparison for a merchant-side system is raw vs. raw + velocity.</p>
        </div>
      </Card>

      <div className="grid cols-3" style={{ marginTop: 16 }}>
        <Card title="Operating point" sub={`Test window at the review threshold ${fmtPct(op.review_threshold, 1)} (validation-chosen)`}>
          <ConfusionMatrix tp={r.tp} fp={r.fp} fn={r.fn} tn={r.tn} />
          <p className="note">Precision {fmtPct(r.precision)} · recall {fmtPct(r.recall)} · {fmtPct(r.alert_rate)} of transactions reviewed.</p>
          {e.protocol.amendments?.map((a) => <p className="note" key={a}><b>Amendment:</b> {a}</p>)}
        </Card>
        <Card title="Calibration" sub="Platt scaling fit on validation; measured on test">
          <table className="data">
            <thead><tr><th /><th className="num">Brier</th><th className="num">Log-loss</th><th className="num">ECE</th></tr></thead>
            <tbody>
              {(["raw", "calibrated"] as const).map((k) => (
                <tr key={k}><td>{k === "raw" ? "Raw model output" : "Calibrated"}</td>
                  <td className="num">{e.calibration_test[k].brier.toFixed(4)}</td>
                  <td className="num">{e.calibration_test[k].log_loss.toFixed(4)}</td>
                  <td className="num">{e.calibration_test[k].ece.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Global feature attribution" sub="Mean |SHAP| on validation; v_* are online velocity features">
          <HBarChart label="Global importance" labelWidth={170}
            rows={e.global_importance.slice(0, 12).map((g) => ({ label: g.feature, value: g.mean_abs_shap }))} valueFormat={(v) => v.toFixed(3)} />
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Evaluation dataset" sub={`${sp.strategy} · IEEE-CIS (Vesta), local only`}>
          <table className="data">
            <thead><tr><th>Split</th><th className="num">Rows</th><th className="num">Frauds</th><th className="num">Fraud rate</th><th className="num">Days</th></tr></thead>
            <tbody>{(["train", "validation", "test"] as const).map((k) => (
              <tr key={k}><td>{k}</td><td className="num">{fmtInt(sp[k].rows)}</td><td className="num">{fmtInt(sp[k].frauds)}</td>
                <td className="num">{fmtPct(sp[k].fraud_rate, 2)}</td><td className="num">{day(sp[k].time_start_s)}–{day(sp[k].time_end_s)}</td></tr>
            ))}</tbody>
          </table>
          <p className="note">Competition data: not redistributed; this page shows aggregate results only.</p>
        </Card>
        <Card title="Model versions" sub="File registry; promotion requires the evaluation gate (or a logged override)">
          <table className="data">
            <thead><tr><th>When</th><th>Action</th><th>Version</th><th>Reason</th></tr></thead>
            <tbody>{(versions?.history ?? []).slice().reverse().map((h) => (
              <tr key={h.at + h.version}><td className="nowrap small">{new Date(h.at).toLocaleString()}</td><td>{h.action}</td>
                <td className="mono">{h.version}{h.version === versions?.serving && <span className="badge info" style={{ marginLeft: 6 }}><span className="dot" />serving</span>}</td>
                <td className="small ink2">{h.reason}</td></tr>
            ))}</tbody>
          </table>
        </Card>
      </div>
    </>
  );
}
