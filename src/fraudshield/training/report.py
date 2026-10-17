"""Machine-readable results + a Markdown report + static figures for a training run."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from fraudshield.models.artifact import write_json

# Validated categorical order (see docs/ARCHITECTURE.md, "Visual language").
SERIES = {
    "xgboost": "#2a78d6",
    "stacked": "#eb6834",
    "logreg": "#1baf7a",  # fixed slot order 1-5
    "isolation_forest": "#eda100",
    "autoencoder": "#e87ba4",
}
LABELS = {
    "xgboost": "XGBoost",
    "stacked": "Stacked ensemble (OOF)",
    "stacked_insample": "Stacked (legacy, in-sample)",
    "logreg": "Logistic regression",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Autoencoder",
    "xgboost_raw_features": "XGBoost (raw features)",
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _fmt(x: Any, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def markdown_report(e: dict[str, Any]) -> str:
    sel = e["selected_model"]
    ds = e["dataset"]
    sp = ds["split"]
    lines = [
        f"# Benchmark report — {e['experiment']}",
        "",
        f"Generated {e['created_at']} by `fraudshield train`. Model version `{e['version']}`.",
        "",
        "## Data and protocol",
        "",
        f"- Source file SHA-256 `{ds['sha256'][:16]}…`, {ds['rows_raw']:,} rows, {ds['frauds_raw']} frauds; "
        f"{ds.get('exact_duplicates_removed', 0):,} exact duplicates removed.",
        f"- Split: **{sp['strategy']}** (out-of-time).",
        "",
        "| Split | Rows | Frauds | Fraud rate | Time window (h) |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("train", "validation", "test"):
        s = sp[name]
        lines.append(
            f"| {name} | {s['rows']:,} | {s['frauds']} | {s['fraud_rate']:.4%} | "
            f"{s['time_start_s'] / 3600:.1f}–{s['time_end_s'] / 3600:.1f} |"
        )
    lines += [
        "",
        "- " + "\n- ".join(f"{k}: {v}" for k, v in e["protocol"].items() if k != "selection_rule"),
        "",
    ]
    lines += (
        ["### Model selection (validation only)", ""]
        + [f"- {r}" for r in e["protocol"]["selection_log"]]
        + [""]
    )

    lines += [
        "## Test results (scored once)",
        "",
        "| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Recall @ FPR 0.1% | Recall @ FPR 1% | Precision @ top 0.1% |",
        "|---|---|---|---:|---:|---:|",
    ]
    ci_pr, ci_roc = e["test_bootstrap_95ci"]["pr_auc"], e["test_bootstrap_95ci"]["roc_auc"]
    for k, m in e["test"].items():
        p_at = next(v for kk, v in m.items() if kk.startswith("precision@top0.10%"))
        star = " **(selected)**" if k == sel else ""
        lines.append(
            f"| {LABELS.get(k, k)}{star} | {m['roc_auc']:.4f} [{ci_roc[k]['ci_low']:.4f}, {ci_roc[k]['ci_high']:.4f}] | "
            f"{m['pr_auc']:.4f} [{ci_pr[k]['ci_low']:.4f}, {ci_pr[k]['ci_high']:.4f}] | "
            f"{m['recall@fpr=0.001']:.3f} | {m['recall@fpr=0.01']:.3f} | {p_at:.3f} |"
        )
    lines += [
        "",
        "Paired bootstrap of test PR-AUC difference vs XGBoost (same resamples):",
        "",
        "| Model | Δ PR-AUC | 95% CI | P(Δ ≤ 0) |",
        "|---|---:|---|---:|",
    ]
    for k, v in ci_pr.items():
        if "delta_vs_xgboost" in v:
            lines.append(
                f"| {LABELS.get(k, k)} | {v['delta_vs_xgboost']:+.4f} | "
                f"[{v['delta_ci_low']:+.4f}, {v['delta_ci_high']:+.4f}] | {v['p_delta_le_0']:.2f} |"
            )

    op = e["selected_operating_points"]
    r, rc, pc = (
        op["review_threshold_test"],
        op["review_threshold_test_recall_ci"],
        op["review_threshold_test_precision_ci"],
    )
    lines += [
        "",
        f"## Operating point of the selected model ({LABELS.get(sel, sel)})",
        "",
        f"Review threshold **{op['review_threshold']:.4f}** (calibrated probability), chosen on validation as the "
        "cost-optimal threshold under the illustrative assumptions below.",
        "",
        "| Metric | Test value |",
        "|---|---:|",
        f"| Precision | {r['precision']:.3f} [{pc['ci_low']:.3f}, {pc['ci_high']:.3f}] |",
        f"| Recall | {r['recall']:.3f} [{rc['ci_low']:.3f}, {rc['ci_high']:.3f}] |",
        f"| F1 | {r['f1']:.3f} |",
        f"| False-positive rate | {r['fpr']:.5f} |",
        f"| Alerts | {r['alerts']:,} of {r['tp'] + r['fp'] + r['tn'] + r['fn']:,} ({r['alert_rate']:.3%}) |",
        f"| Confusion matrix | TP {r['tp']} · FP {r['fp']} · FN {r['fn']} · TN {r['tn']:,} |",
        "",
    ]
    cal = e["calibration_test"]
    lines += [
        "## Calibration (test)",
        "",
        "| | Brier | Log-loss | ECE |",
        "|---|---:|---:|---:|",
        f"| Raw model output | {cal['raw']['brier']:.6f} | {cal['raw']['log_loss']:.5f} | {cal['raw']['ece']:.5f} |",
        f"| Platt-calibrated | {cal['calibrated']['brier']:.6f} | {cal['calibrated']['log_loss']:.5f} | "
        f"{cal['calibrated']['ece']:.5f} |",
        "",
    ]
    ce = e["cost_experiment"]
    lines += [
        "## Cost experiment (illustrative assumptions)",
        "",
        f"Assumptions: review cost €{ce['assumptions']['review_cost']:.2f} per alert; a missed fraud loses "
        f"{ce['assumptions']['loss_rate']:.0%} of its amount; reviewed fraud is caught. {ce['assumptions']['note']}",
        "",
        "| Policy on test window | Reviews | Frauds caught | Fraud loss (€) | Review cost (€) | Total (€) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in ce["test"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {row.get('reviews', 0):,} | {row.get('frauds_caught', '—')} | "
            f"{row.get('fraud_loss', row['total_cost'] if name == 'approve_all' else 0):,.2f} | "
            f"{row.get('review_cost', row['total_cost'] if name == 'review_all' else 0):,.2f} | "
            f"{row['total_cost']:,.2f} |"
        )
    lines += [
        "",
        "The *test oracle* row uses test labels to pick the threshold and is shown only as an upper bound.",
        "",
    ]
    lines += [
        "## In-process latency (single transaction, this machine)",
        "",
        "| Model | p50 (ms) | p95 (ms) | p99 (ms) | Batch throughput (rows/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in e["latency_in_process"].items():
        lines.append(
            f"| {LABELS.get(k, k)} | {v['single_row_p50_ms']:.3f} | {v['single_row_p95_ms']:.3f} | "
            f"{v['single_row_p99_ms']:.3f} | {v['batch_rows_per_s']:,.0f} |"
        )
    lines += [
        "",
        "## Global feature attribution (mean |SHAP|, validation)",
        "",
        "| Feature | mean abs SHAP (log-odds) |",
        "|---|---:|",
    ]
    lines += [f"| {g['feature']} | {g['mean_abs_shap']:.4f} |" for g in e["global_importance"][:12]]
    lines += [
        "",
        "SHAP values describe how the model uses features; they are not causal evidence of fraud.",
        "",
    ]
    env = e["environment"]
    lines += [
        "## Environment",
        "",
        ", ".join(f"{k} `{v}`" for k, v in env.items()),
        "",
        f"Runtime: {e['runtime_seconds']} s.",
        "",
    ]
    return "\n".join(lines)


def _style(ax: Any) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def write_figures(e: dict[str, Any], fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})

    for curve, xl, yl, fname in (
        ("pr", "Recall", "Precision", "pr_curves.png"),
        ("roc", "False-positive rate", "True-positive rate", "roc_curves.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
        for k, color in SERIES.items():
            pts = e["curves_test"][k][curve]
            auc = e["test"][k]["pr_auc" if curve == "pr" else "roc_auc"]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                linewidth=2,
                label=f"{LABELS[k]} ({'AP' if curve == 'pr' else 'AUC'} {auc:.3f})",
            )
        if curve == "roc":
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(
            f"{'Precision-recall' if curve == 'pr' else 'ROC (log FPR)'} — test split", color=INK, loc="left"
        )
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / fname)
        plt.close(fig)
        written.append(fname)

    cal = e["calibration_test"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.plot([1e-4, 1], [1e-4, 1], color=MUTED, linewidth=1)
    for key, color, lab in (
        ("raw", "#eb6834", "Raw model output"),
        ("calibrated", "#2a78d6", "Platt-calibrated"),
    ):
        b = [x for x in cal[key]["reliability"] if x["n"] >= 20]
        ax.plot(
            [x["mean_predicted"] for x in b],
            [max(x["observed_rate"], 1e-4) for x in b],
            marker="o",
            markersize=5,
            color=color,
            linewidth=2,
            label=lab,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraud rate (bin)")
    ax.set_title("Reliability — test split", color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png")
    plt.close(fig)
    written.append("calibration.png")

    ce = e["cost_experiment"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for key, color, lab in (("validation_curve", "#eb6834", "Validation"), ("test_curve", "#2a78d6", "Test")):
        rows = [r for r in ce[key] if r["threshold"] >= 1e-3]  # below 0.1% every policy reviews most traffic
        ax.plot(
            [r["threshold"] for r in rows],
            [r["total_cost"] for r in rows],
            color=color,
            linewidth=2,
            label=lab,
        )
    ax.axvline(ce["validation_cost_optimal_threshold"], color=MUTED, linewidth=1)
    ax.annotate(
        "threshold chosen\non validation",
        (ce["validation_cost_optimal_threshold"], ax.get_ylim()[1] * 0.9),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Review threshold (calibrated probability)")
    ax.set_ylabel("Expected cost (€, illustrative)")
    ax.set_title("Operational cost vs. review threshold", color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cost_curve.png")
    plt.close(fig)
    written.append("cost_curve.png")

    gi = e["global_importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=150)
    ax.barh([g["feature"] for g in gi], [g["mean_abs_shap"] for g in gi], color="#2a78d6", height=0.6)
    ax.set_xlabel("Mean |SHAP| (calibrated log-odds)")
    ax.set_title(f"Global attribution — {LABELS.get(e['selected_model'])}", color=INK, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "global_importance.png")
    plt.close(fig)
    written.append("global_importance.png")
    return written


def write_report(evaluation: dict[str, Any], out_dir: Path, make_figures: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", evaluation)
    (out_dir / "report.md").write_text(markdown_report(evaluation), encoding="utf-8")
    if make_figures:
        with contextlib.suppress(ImportError):  # matplotlib is a training extra
            write_figures(evaluation, out_dir / "figures")
