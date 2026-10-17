import { api } from "../lib/api";
import { MODEL_LABEL, fmtEur, fmtInt, fmtMs, fmtPct } from "../lib/format";
import { usePoll } from "../lib/hooks";
import { ConfusionMatrix, CurveChart, HBarChart } from "../components/charts";
import { Card, ErrorNote, PageHead } from "../components/ui";

// Fixed categorical slots: colour follows the model, never its rank.
const SLOT: Record<string, string> = {
  xgboost: "var(--s1)", stacked: "var(--s2)", logreg: "var(--s3)", isolation_forest: "var(--s4)", autoencoder: "var(--s5)",
};
const CURVE_MODELS = ["xgboost", "stacked", "logreg", "isolation_forest", "autoencoder"];
const TABLE_MODELS = ["xgboost", "stacked", "logreg", "xgboost_raw_features", "stacked_insample", "isolation_forest", "autoencoder"];

export default function Performance() {
  const ev = usePoll(api.evaluation, 0);
  const versions = usePoll(api.versions, 0);
  const e = ev.data;
  if (!e) return <><PageHead title="Model performance" /><ErrorNote error={ev.error} />{!ev.error && <div className="empty">Loading evaluation…</div>}</>;

  const sel = e.selected_model;
  const op = e.selected_operating_points;
  const r = op.review_threshold_test;
  const sp = e.dataset.split;
  const ci = e.test_bootstrap_95ci;
  const recallKey = (m: Record<string, number>, prefix: string) => Object.keys(m).find((k) => k.startsWith(prefix)) ?? "";
  const lat = e.latency_in_process;
  const cost = e.cost_experiment;
  const visible = (rows: { threshold: number; total_cost: number }[]) => rows.filter((p) => p.threshold >= 1e-3);

  return (
    <>
      <PageHead title="Model performance"
        desc={<>Offline evaluation stored with model <span className="mono">{e.version}</span>, created {new Date(e.created_at).toLocaleString()}.
          Out-of-time split; every choice was made on validation and the test window was scored once.</>} />

      <div className="kpis">
        {[
          ["Selected model", MODEL_LABEL[sel] ?? sel, "pre-registered rule, validation PR-AUC"],
          ["Test PR-AUC", e.test[sel].pr_auc.toFixed(3), `95% CI ${ci.pr_auc[sel].ci_low.toFixed(3)}–${ci.pr_auc[sel].ci_high.toFixed(3)}`],
          ["Test ROC-AUC", e.test[sel].roc_auc.toFixed(4), `95% CI ${ci.roc_auc[sel].ci_low.toFixed(3)}–${ci.roc_auc[sel].ci_high.toFixed(3)}`],
          ["Recall @ review threshold", fmtPct(r.recall), `CI ${fmtPct(op.review_threshold_test_recall_ci.ci_low)}–${fmtPct(op.review_threshold_test_recall_ci.ci_high)} · ${r.tp}/${r.tp + r.fn} frauds`],
          ["Precision @ review threshold", fmtPct(r.precision), `${fmtInt(r.alerts)} alerts · FPR ${fmtPct(r.fpr, 3)}`],
          ["Single-row latency (p95)", fmtMs(lat[sel]?.single_row_p95_ms), "in-process, this machine"],
        ].map(([l, v, s]) => (
          <div className="card stat" key={l}><div className="label">{l}</div><div className="value" style={{ fontSize: 21 }}>{v}</div><div className="sub">{s}</div></div>
        ))}
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Precision–recall (test)" sub="The informative curve when fraud is ~0.1% of traffic">
          <CurveChart label="Precision-recall curves" xLabel="Recall" yLabel="Precision"
            curves={CURVE_MODELS.map((k) => ({ key: k, label: `${MODEL_LABEL[k]} · AP ${e.test[k].pr_auc.toFixed(3)}`, color: SLOT[k], points: e.curves_test[k].pr }))} />
        </Card>
        <Card title="ROC (test, log false-positive rate)" sub="Log scale shows the low-FPR region where fraud operations live">
          <CurveChart label="ROC curves" xLabel="False-positive rate" yLabel="True-positive rate" xLog xDomain={[1e-4, 1]}
            xFormat={(v) => (v >= 0.01 ? `${v * 100}%` : `${+(v * 100).toPrecision(1)}%`)}
            curves={CURVE_MODELS.map((k) => ({ key: k, label: `${MODEL_LABEL[k]} · AUC ${e.test[k].roc_auc.toFixed(3)}`, color: SLOT[k], points: e.curves_test[k].roc }))} />
        </Card>
      </div>

      <Card title="Baselines vs. ensemble" sub="Validation drove selection; test shown with 95% bootstrap CIs and paired differences vs. XGBoost" flush className="mt">
        <div className="table-wrap">
          <table className="data">
            <thead><tr>
              <th>Model</th><th className="num">Val PR-AUC</th><th className="num">Test PR-AUC [95% CI]</th><th className="num">Δ vs XGBoost [95% CI]</th>
              <th className="num">Test ROC-AUC</th><th className="num">Recall @ FPR 0.1%</th><th className="num">Precision @ top 0.1%</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {TABLE_MODELS.filter((k) => e.test[k]).map((k) => {
                const c = ci.pr_auc[k];
                return (
                  <tr key={k} style={k === sel ? { background: "var(--accent-wash)" } : undefined}>
                    <td className="nowrap"><b>{MODEL_LABEL[k] ?? k}</b>{k === sel && <span className="badge info" style={{ marginLeft: 8 }}><span className="dot" />selected</span>}</td>
                    <td className="num">{e.validation[k].pr_auc.toFixed(3)}</td>
                    <td className="num nowrap">{e.test[k].pr_auc.toFixed(3)} <span className="muted">[{c.ci_low.toFixed(3)}, {c.ci_high.toFixed(3)}]</span></td>
                    <td className="num nowrap">{c.delta_vs_xgboost == null ? "—" : <>{c.delta_vs_xgboost >= 0 ? "+" : ""}{c.delta_vs_xgboost.toFixed(3)} <span className="muted">[{c.delta_ci_low!.toFixed(3)}, {c.delta_ci_high!.toFixed(3)}]</span></>}</td>
                    <td className="num">{e.test[k].roc_auc.toFixed(4)}</td>
                    <td className="num">{fmtPct(e.test[k][recallKey(e.test[k], "recall@fpr=0.001")])}</td>
                    <td className="num">{fmtPct(e.test[k][recallKey(e.test[k], "precision@top0.10%")])}</td>
                    <td className="small ink2">{e.candidates[k]?.notes}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ padding: "4px 16px 14px" }}>
          <p className="note"><b>Selection log:</b> {e.protocol.selection_log.join(" · ")}</p>
        </div>
      </Card>

      <div className="grid cols-3" style={{ marginTop: 16 }}>
        <Card title="Confusion matrix" sub={`Test window at the review threshold ${fmtPct(op.review_threshold, 2)} (chosen on validation)`}>
          <ConfusionMatrix tp={r.tp} fp={r.fp} fn={r.fn} tn={r.tn} />
          <p className="note">Precision {fmtPct(r.precision)} · recall {fmtPct(r.recall)} · F1 {r.f1.toFixed(3)}.</p>
        </Card>
        <Card title="Calibration" sub="Platt scaling fit on validation; measured on test">
          <table className="data">
            <thead><tr><th /><th className="num">Brier</th><th className="num">Log-loss</th><th className="num">ECE</th></tr></thead>
            <tbody>
              {(["raw", "calibrated"] as const).map((k) => (
                <tr key={k}><td>{k === "raw" ? "Raw model output" : "Calibrated"}</td>
                  <td className="num">{e.calibration_test[k].brier.toFixed(5)}</td>
                  <td className="num">{e.calibration_test[k].log_loss.toFixed(4)}</td>
                  <td className="num">{e.calibration_test[k].ece.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
          <table className="data" style={{ marginTop: 10 }}>
            <thead><tr><th>Predicted bin</th><th className="num">Txns</th><th className="num">Mean predicted</th><th className="num">Observed</th></tr></thead>
            <tbody>{e.calibration_test.calibrated.reliability.map((b) => (
              <tr key={b.lo}><td className="nowrap">{fmtPct(b.lo)}–{fmtPct(b.hi)}</td><td className="num">{fmtInt(b.n)}</td>
                <td className="num">{fmtPct(b.mean_predicted, 2)}</td><td className="num">{fmtPct(b.observed_rate, 2)} <span className="muted small">({b.positives})</span></td></tr>
            ))}</tbody>
          </table>
        </Card>
        <Card title="Latency by model" sub="In-process single-transaction scoring (features + model + calibration)">
          <HBarChart label="p95 latency by model" labelWidth={130} valueFormat={(v) => fmtMs(v)}
            rows={Object.entries(lat).map(([k, v]) => ({ label: MODEL_LABEL[k] ?? k, value: v.single_row_p95_ms,
              detail: `p50 ${fmtMs(v.single_row_p50_ms)} · p99 ${fmtMs(v.single_row_p99_ms)} · batch ${fmtInt(v.batch_rows_per_s)} rows/s` }))} />
          <p className="note">The stacked ensemble pays for its Isolation Forest + autoencoder at every request.</p>
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Operational cost vs. review threshold" sub={`Illustrative: €${cost.assumptions.review_cost} per review, missed fraud loses its amount`}>
          <CurveChart label="Expected cost by threshold" xLabel="Review threshold (calibrated probability)" yLabel="Cost (€)" xLog
            xDomain={[1e-3, 0.99]} yDomain={[0, Math.max(...visible(cost.test_curve).map((p) => p.total_cost), ...visible(cost.validation_curve).map((p) => p.total_cost)) * 1.05]}
            xFormat={(v) => fmtPct(v, v < 0.01 ? 2 : 0)} yFormat={(v) => `€${Math.round(v).toLocaleString()}`}
            markers={[{ x: cost.validation_cost_optimal_threshold, label: "chosen on validation" }]}
            curves={[
              { key: "val", label: "Validation window", color: "var(--s2)", points: visible(cost.validation_curve).map((p) => [p.threshold, p.total_cost]) },
              { key: "test", label: "Test window", color: "var(--s1)", points: visible(cost.test_curve).map((p) => [p.threshold, p.total_cost]) },
            ]} />
          <table className="data" style={{ marginTop: 8 }}>
            <thead><tr><th>Policy on test window</th><th className="num">Reviews</th><th className="num">Total cost</th></tr></thead>
            <tbody>{Object.entries(cost.test).map(([k, v]) => (
              <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="num">{fmtInt(v.reviews ?? 0)}</td><td className="num">{fmtEur(v.total_cost, true)}</td></tr>
            ))}</tbody>
          </table>
          <p className="note">“Test oracle” uses test labels to pick its threshold and is only an upper bound.</p>
        </Card>
        <Card title="Global feature attribution" sub="Mean |SHAP| on validation (calibrated log-odds)">
          <HBarChart label="Global importance" labelWidth={110}
            rows={e.global_importance.slice(0, 14).map((g) => ({ label: g.feature, value: g.mean_abs_shap }))} valueFormat={(v) => v.toFixed(3)} />
          <p className="note">Importance reflects how this model uses inputs, not causal drivers of fraud.</p>
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Card title="Evaluation dataset" sub={`${sp.strategy} · ${e.dataset.time_span_hours} h capture`}>
          <table className="data">
            <thead><tr><th>Split</th><th className="num">Rows</th><th className="num">Frauds</th><th className="num">Fraud rate</th><th className="num">Hours</th></tr></thead>
            <tbody>{(["train", "validation", "test"] as const).map((k) => (
              <tr key={k}><td>{k}</td><td className="num">{fmtInt(sp[k].rows)}</td><td className="num">{sp[k].frauds}</td>
                <td className="num">{fmtPct(sp[k].fraud_rate, 3)}</td>
                <td className="num">{(sp[k].time_start_s / 3600).toFixed(1)}–{(sp[k].time_end_s / 3600).toFixed(1)}</td></tr>
            ))}</tbody>
          </table>
          <p className="note">ULB credit-card dataset (Sept 2013, 2 days). SHA-256 <span className="mono">{e.dataset.sha256.slice(0, 16)}…</span>;
            {" "}{fmtInt(e.dataset.exact_duplicates_removed)} exact duplicates removed before splitting.</p>
        </Card>
        <Card title="Model versions" sub="File registry; promotion requires the evaluation gate">
          <table className="data">
            <thead><tr><th>When</th><th>Action</th><th>Version</th></tr></thead>
            <tbody>{(versions.data?.history ?? []).slice().reverse().map((h) => (
              <tr key={h.at + h.version}><td className="nowrap small">{new Date(h.at).toLocaleString()}</td><td>{h.action}</td>
                <td className="mono">{h.version}{h.version === versions.data?.serving && <span className="badge info" style={{ marginLeft: 6 }}><span className="dot" />serving</span>}</td></tr>
            ))}</tbody>
          </table>
          <p className="note">Environment: {Object.entries(e.environment).filter(([k]) => ["python", "xgboost", "sklearn", "git_commit"].includes(k)).map(([k, v]) => `${k} ${v}`).join(" · ")}</p>
        </Card>
      </div>
    </>
  );
}
