"""FraudShield v2: train and evaluate the IEEE-CIS model on authorisation-time fields.

Protocol (pre-registered in configs/ieee_v2.yaml, results in docs/ML_EVALUATION.md §12):
1. Load only the field contract; compute velocity features offline over the time-ordered history
   (past-only, the same definitions the online store serves).
2. Chronological 60/20/20 split. Early stopping uses the end of the training period.
3. Two variants: raw contract vs raw + velocity. Velocity is adopted only if its validation PR-AUC
   gain is positive with a paired-bootstrap 90% CI above zero.
4. Calibration (Platt) and thresholds on validation; test scored once; the all-431-column model
   from benchmarks/results/ieee-cis is the reference ceiling.
5. Rolling monthly backtest with a label-maturity gap: a static model vs monthly retraining.
6. A standard, checksummed artifact for the "ieee" serving profile (+ replay and warm-up files).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from fraudshield import __version__
from fraudshield.data.ieee_cis import load_contract
from fraudshield.decision_engine.simulate import capacity_curve
from fraudshield.features.velocity import DAY, VELOCITY_SPECS, OnlineFeatureStore, compute_offline
from fraudshield.models.artifact import ModelRegistry, save_model, write_json
from fraudshield.models.calibration import PlattCalibrator
from fraudshield.models.fraud_model import FraudModel
from fraudshield.monitoring.drift import build_reference
from fraudshield.profiles.ieee import AMOUNT, CONTRACT_FIELDS, LABEL, TIME, IEEEPreprocessor
from fraudshield.training import evaluate as ev
from fraudshield.training.pipeline import _git_commit, _log, booster_proba

REPLAY_COLUMNS = ["TransactionID", TIME, *CONTRACT_FIELDS, LABEL]
MAX_WINDOW_S = max(s.window_s for s in VELOCITY_SPECS)


def chrono_fit_xgb(
    x: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[xgb.Booster, int]:
    """Fit with early stopping on the *last* ``early_stopping_frac`` of the (time-ordered) rows."""
    p = dict(params)
    es_frac = float(p.pop("early_stopping_frac"))
    cut = int(len(y) * (1 - es_frac))
    clf = xgb.XGBClassifier(random_state=seed, n_jobs=-1, **p)
    clf.fit(x[:cut], y[:cut], eval_set=[(x[cut:], y[cut:])], verbose=False)
    best = int(clf.best_iteration)
    return clf.get_booster()[: best + 1], best


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [c for c in [TIME, *CONTRACT_FIELDS] if c in frame]
    return frame[cols].to_dict("records")


def _data_hash(root: Path | None, frame: pd.DataFrame) -> str:
    if root is not None and (root / "train_transaction.csv").exists():
        h = hashlib.sha256()
        with open(root / "train_transaction.csv", "rb") as f:
            while block := f.read(1 << 20):
                h.update(block)
        return h.hexdigest()
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def measure_online_latency(
    model: FraudModel, history: pd.DataFrame, rows: pd.DataFrame, n: int, velocity: bool
) -> dict[str, float]:
    """End-to-end per-transaction time: online feature lookup + update, preprocessing, model."""
    store = OnlineFeatureStore()
    store.warm_up(_records(history))
    model.prepare_for_serving()
    recs = _records(rows.iloc[:n])
    feat_ms, total_ms = [], []
    for rec in recs:
        t0 = time.perf_counter()
        feats = store.score_and_update(rec) if velocity else {}
        t1 = time.perf_counter()
        x = model.base_features({k: [v] for k, v in {**rec, **feats}.items()})
        model.predict_from_inputs(model.model_inputs(x))
        t2 = time.perf_counter()
        feat_ms.append((t1 - t0) * 1000)
        total_ms.append((t2 - t0) * 1000)
    a, f = np.asarray(total_ms[10:] or total_ms), np.asarray(feat_ms[10:] or feat_ms)
    return {
        "single_row_p50_ms": float(np.percentile(a, 50)),
        "single_row_p95_ms": float(np.percentile(a, 95)),
        "single_row_p99_ms": float(np.percentile(a, 99)),
        "feature_store_p50_ms": float(np.percentile(f, 50)),
        "feature_store_p95_ms": float(np.percentile(f, 95)),
        "rows": len(a),
        "warm_up_events": len(history),
    }


def backtest(
    df: pd.DataFrame, velocity: bool, params: dict[str, Any], cfg: dict[str, Any], seed: int
) -> list[dict[str, Any]]:
    """Rolling monthly backtest: static model vs retraining with a label-maturity gap."""
    bt = cfg["backtest"]
    block = ((df[TIME] - df[TIME].min()) // (bt["block_days"] * DAY)).to_numpy()
    y = df[LABEL].to_numpy()

    def fit_on(mask: np.ndarray) -> tuple[IEEEPreprocessor, xgb.Booster]:
        prep = IEEEPreprocessor(velocity=velocity).fit(df[mask])
        booster, _ = chrono_fit_xgb(prep.transform(df[mask]), y[mask], params, seed)
        return prep, booster

    static_prep, static_booster = fit_on(block < bt["static_train_blocks"])
    rows = []
    for m in bt["test_blocks"]:
        test = block == m
        if test.sum() == 0 or y[test].sum() == 0:
            continue
        train = block < m - bt["label_maturity_gap_blocks"]
        prep, booster = fit_on(train)
        p_static = booster_proba(static_booster, static_prep.transform(df[test]))
        p_retrained = booster_proba(booster, prep.transform(df[test]))
        rows.append(
            {
                "test_block": int(m),
                "train_blocks_retrained": f"0-{int(m - bt['label_maturity_gap_blocks'] - 1)}",
                "train_blocks_static": f"0-{bt['static_train_blocks'] - 1}",
                "rows": int(test.sum()),
                "frauds": int(y[test].sum()),
                "static_pr_auc": float(average_precision_score(y[test], p_static)),
                "retrained_pr_auc": float(average_precision_score(y[test], p_retrained)),
                "static_recall@fpr=0.01": ev.recall_at_fpr(y[test], p_static, 0.01),
                "retrained_recall@fpr=0.01": ev.recall_at_fpr(y[test], p_retrained, 0.01),
            }
        )
        _log(
            f"backtest block {m}: static {rows[-1]['static_pr_auc']:.4f} retrained {rows[-1]['retrained_pr_auc']:.4f}"
        )
    return rows


def run_ieee_v2(
    config: dict[str, Any],
    root: str | Path | None = None,
    frame: pd.DataFrame | None = None,
    artifacts_root: str | Path = "artifacts/ieee",
    results_dir: str | Path | None = None,
    promote: bool = True,
    quick: bool = False,
    force_variant: str | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg = json.loads(json.dumps(config))
    seed = int(cfg["seed"])
    params = dict(cfg["xgboost"])
    if quick:  # tests / CI on synthetic data
        params.update(n_estimators=150, early_stopping_rounds=20)
        cfg["selection"]["bootstrap"] = cfg["evaluation"]["bootstrap"] = 30
        cfg["evaluation"]["latency_rows"] = 40
        cfg["backtest"].update(block_days=5, static_train_blocks=2, test_blocks=[3, 4, 5])

    root_path = Path(root) if root is not None else None
    df = frame.copy() if frame is not None else load_contract(root_path or cfg["data"]["root"])
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    _log(f"loaded {len(df)} rows; computing {len(VELOCITY_SPECS)} velocity features offline")
    df = pd.concat([df, compute_offline(df)], axis=1)

    n = len(df)
    a = int(n * cfg["split"]["train_frac"])
    b = int(n * (cfg["split"]["train_frac"] + cfg["split"]["val_frac"]))
    train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    y_tr, y_va, y_te = (p[LABEL].to_numpy() for p in (train, val, test))

    variants: dict[str, dict[str, Any]] = {}
    for name, velocity in (("raw_contract", False), ("raw_plus_velocity", True)):
        prep = IEEEPreprocessor(velocity=velocity).fit(train)
        x_tr, x_va, x_te = prep.transform(train), prep.transform(val), prep.transform(test)
        booster, best = chrono_fit_xgb(x_tr, y_tr, params, seed)
        variants[name] = {
            "prep": prep,
            "booster": booster,
            "trees": best + 1,
            "x_tr": x_tr,
            "x_va": x_va,
            "pv": booster_proba(booster, x_va),
            "pt": booster_proba(booster, x_te),
        }
        _log(
            f"{name}: {len(prep.feature_names)} inputs, {best + 1} trees, "
            f"val PR-AUC {average_precision_score(y_va, variants[name]['pv']):.4f}"
        )

    sel_boot = ev.bootstrap(
        y_va,
        {k: v["pv"] for k, v in variants.items()},
        ev.average_precision_score,
        n_boot=int(cfg["selection"]["bootstrap"]),
        seed=seed,
        reference="raw_contract",
        alpha=float(cfg["selection"]["alpha"]),
    )
    gain = sel_boot["raw_plus_velocity"]["delta_vs_raw_contract"]
    ci_low = sel_boot["raw_plus_velocity"]["delta_ci_low"]
    selected = "raw_plus_velocity" if gain > 0 and ci_low > 0 else "raw_contract"
    selection_log = (
        f"velocity: val PR-AUC gain {gain:+.4f} (90% CI low {ci_low:+.4f}) -> selected: {selected}"
    )
    if force_variant is not None:  # tests / ablations only; the override is recorded in the report
        selected = force_variant
        selection_log += f"; OVERRIDDEN by force_variant={force_variant}"
    _log(selection_log)

    v = variants[selected]
    calibrator = PlattCalibrator().fit(v["pv"], y_va)
    p_va, p_te = calibrator.transform(v["pv"]), calibrator.transform(v["pt"])
    costs = cfg["cost_assumptions"]
    amt_va, amt_te = val[AMOUNT].to_numpy(), test[AMOUNT].to_numpy()
    val_curve = ev.review_cost_curve(y_va, p_va, amt_va, costs["review_cost"], costs["loss_rate"])
    uncapped_threshold = min(val_curve, key=lambda r: r["total_cost"])["threshold"]
    max_rate = float(cfg.get("operating_point", {}).get("max_review_rate", 1.0))
    feasible = [r for r in val_curve if r["reviews"] <= max_rate * len(y_va)]
    review_threshold = (
        min(feasible, key=lambda r: r["total_cost"])["threshold"]
        if feasible
        else max(r["threshold"] for r in val_curve)
    )
    f1_threshold = ev.best_f1_threshold(y_va, p_va)

    _log("scoring test window (final report)")
    test_metrics = {k: ev.ranking_metrics(y_te, x["pt"]) for k, x in variants.items()}
    n_boot = int(cfg["evaluation"]["bootstrap"])
    ci = ev.bootstrap(
        y_te,
        {k: x["pt"] for k, x in variants.items()},
        ev.average_precision_score,
        n_boot=n_boot,
        seed=seed,
        reference="raw_contract",
    )
    ceiling = None
    ref_file = Path(cfg.get("reference_full_feature_pr_auc_file", ""))
    if not quick and ref_file.is_file():
        ceiling = json.loads(ref_file.read_text())["variants"]["none"]["test"]["pr_auc"]
    raw_pr, vel_pr = test_metrics["raw_contract"]["pr_auc"], test_metrics["raw_plus_velocity"]["pr_auc"]
    gap = (vel_pr - raw_pr) / (ceiling - raw_pr) if ceiling is not None and ceiling > raw_pr else None

    hours = float(test[TIME].max() - test[TIME].min()) / 3600
    op = ev.threshold_metrics(y_te, p_te, review_threshold)
    model = FraudModel("xgboost", v["prep"], booster=v["booster"], calibrator=calibrator)
    history = df[(df[TIME] >= test[TIME].min() - MAX_WINDOW_S) & (df.index < b)]
    latency = measure_online_latency(
        model, history, test, int(cfg["evaluation"]["latency_rows"]), velocity=selected == "raw_plus_velocity"
    )
    _log(
        f"online latency p50 {latency['single_row_p50_ms']:.3f} ms p95 {latency['single_row_p95_ms']:.3f} ms"
    )

    sample = v["x_va"][: min(20_000, len(v["x_va"]))]
    contrib, _ = model.contributions(sample)
    imp = np.abs(contrib).mean(axis=0)
    global_importance = [
        {"feature": model.model_feature_names[i], "mean_abs_shap": float(imp[i])}
        for i in np.argsort(-imp)[:25]
    ]

    _log("rolling monthly backtest")
    bt_rows = backtest(df, selected == "raw_plus_velocity", params, cfg, seed)

    created = datetime.now(UTC)
    version = f"ieee-v2-{selected.replace('_', '-')}-{created:%Y%m%d-%H%M%S}"
    split = {
        "strategy": f"chronological {cfg['split']['train_frac']:.0%}/{cfg['split']['val_frac']:.0%}/"
        f"{1 - cfg['split']['train_frac'] - cfg['split']['val_frac']:.0%}",
        **{
            k: {
                "rows": len(p),
                "frauds": int(p[LABEL].sum()),
                "fraud_rate": round(float(p[LABEL].mean()), 6),
                "time_start_s": float(p[TIME].min()),
                "time_end_s": float(p[TIME].max()),
            }
            for k, p in (("train", train), ("validation", val), ("test", test))
        },
    }
    evaluation: dict[str, Any] = {
        "version": version,
        "experiment": cfg["experiment_name"],
        "profile": "ieee",
        "created_at": created.isoformat(timespec="seconds"),
        "dataset": {
            "name": "IEEE-CIS train_transaction + train_identity (field contract only)",
            "rows": n,
            "frauds": int(df[LABEL].sum()),
            "split": split,
        },
        "protocol": {
            "field_contract": list(CONTRACT_FIELDS),
            "velocity_features": [
                {
                    "name": s.name,
                    "entity": list(s.entity),
                    "window_s": s.window_s,
                    "agg": s.agg,
                    "field": s.field,
                }
                for s in VELOCITY_SPECS
            ],
            "selection_rule": "velocity adopted iff validation PR-AUC gain > 0 with paired-bootstrap 90% CI > 0",
            "selection_log": [selection_log],
            "early_stopping": "last 15% of the training period",
            "calibration": "Platt scaling on validation",
            "review_threshold_rule": f"cost-optimal on validation among thresholds flagging <= {max_rate:.0%} "
            "of validation transactions",
            "test_usage": "scored once after all choices were frozen",
            "amendments": [
                "2026-10-17: review threshold capped at a 5% validation review rate after the first full "
                f"run (uncapped cost-optimal threshold {uncapped_threshold:.3f}). Validation-only rule, but "
                "changed after a test run: test operating-point numbers are post-hoc."
            ]
            if max_rate < 1.0
            else [],
        },
        "selected_model": selected,
        "candidates": {
            k: {
                "family": "supervised",
                "notes": "",
                "info": {"trees": x["trees"], "inputs": len(x["prep"].feature_names)},
            }
            for k, x in variants.items()
        },
        "validation": {k: ev.ranking_metrics(y_va, x["pv"]) for k, x in variants.items()},
        "validation_bootstrap_pr_auc_90ci": sel_boot,
        "test": {
            k: {
                **m,
                "at_val_f1_threshold": ev.threshold_metrics(
                    y_te, variants[k]["pt"], ev.best_f1_threshold(y_va, variants[k]["pv"])
                ),
            }
            for k, m in test_metrics.items()
        },
        "test_bootstrap_95ci": {"pr_auc": ci},
        "reference_all_431_columns_test_pr_auc": ceiling,
        "share_of_gap_to_vendor_features_recovered": gap,
        "selected_operating_points": {
            "review_threshold": review_threshold,
            "review_threshold_test": op,
            "f1_threshold": f1_threshold,
            "f1_threshold_test": ev.threshold_metrics(y_te, p_te, f1_threshold),
        },
        "calibration_test": {
            "raw": ev.calibration_metrics(y_te, v["pt"]),
            "calibrated": ev.calibration_metrics(y_te, p_te),
        },
        "capacity_test": capacity_curve(
            y_te, p_te, amt_te, hours, [10, 25, 50, 100, 200], costs["loss_rate"]
        ),
        "latency_online": latency,
        "global_importance": global_importance,
        "curves_test": {k: ev.curves(y_te, x["pt"]) for k, x in variants.items()},
        "backtest": {"settings": cfg["backtest"], "variant": selected, "blocks": bt_rows},
        "environment": {
            "fraudshield": __version__,
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "git_commit": _git_commit(),
        },
        "runtime_seconds": None,
    }
    gate = cfg["gate"]
    sel_test = test_metrics[selected]
    metadata = {
        "version": version,
        "kind": "xgboost",
        "profile": "ieee",
        "selected_candidate": selected,
        "created_at": evaluation["created_at"],
        "data_sha256": _data_hash(
            root_path or (None if frame is not None else Path(cfg["data"]["root"])), df[REPLAY_COLUMNS]
        ),
        "split": split,
        "split_config": cfg["split"],
        "feature_names": v["prep"].feature_names,
        "model_feature_names": v["prep"].feature_names,
        "thresholds": {"review": review_threshold, "f1": f1_threshold},
        "metrics": {
            "validation": evaluation["validation"][selected],
            "test": {k: x for k, x in sel_test.items()},
        },
        "gate_checks": {
            "test_pr_auc": [
                sel_test["pr_auc"],
                ">=",
                gate["min_test_pr_auc"],
                sel_test["pr_auc"] >= gate["min_test_pr_auc"],
            ],
            "test_roc_auc": [
                sel_test["roc_auc"],
                ">=",
                gate["min_test_roc_auc"],
                sel_test["roc_auc"] >= gate["min_test_roc_auc"],
            ],
            "p95_latency_ms": [
                latency["single_row_p95_ms"],
                "<=",
                gate["max_p95_latency_ms"],
                latency["single_row_p95_ms"] <= gate["max_p95_latency_ms"],
            ],
        },
        "velocity_specs": [s.name for s in VELOCITY_SPECS],
        "environment": evaluation["environment"],
    }
    reference = build_reference(
        v["x_tr"], v["prep"].feature_names, p_va, alert_rate=float((p_va >= review_threshold).mean())
    )
    registry = ModelRegistry(artifacts_root)
    model_dir = save_model(model, registry.models_dir / version, metadata, reference, evaluation)
    _log(f"saved model artifact {model_dir}")
    if promote:
        from fraudshield.training.promotion import PromotionRejected, promote_version

        try:
            report = promote_version(registry, version, gate, reason="trained by ieee_v2 pipeline")
            evaluation["promotion"] = {"promoted": True, "report": report}
        except PromotionRejected as exc:  # a failed gate is a result, not a crash
            evaluation["promotion"] = {"promoted": False, "report": exc.report}
            failed = [c["name"] for c in exc.report["checks"] if not c["passed"]]
            _log(f"NOT promoted: gate failed on {', '.join(failed)}")

    replay_dir = Path(artifacts_root) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    test[REPLAY_COLUMNS].to_csv(replay_dir / "ieee_test_window.csv.gz", index=False, compression="gzip")
    history[REPLAY_COLUMNS].to_csv(replay_dir / "ieee_warmup.csv.gz", index=False, compression="gzip")

    evaluation["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    if results_dir is not None:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "metrics.json", evaluation)
        (out / "report.md").write_text(markdown_report(evaluation), encoding="utf-8")
    _log(f"done in {evaluation['runtime_seconds']} s")
    return {"version": version, "model_dir": str(model_dir), "evaluation": evaluation}


def markdown_report(e: dict[str, Any]) -> str:
    sp = e["dataset"]["split"]
    t, ci = e["test"], e["test_bootstrap_95ci"]["pr_auc"]
    lines = [
        f"# FraudShield v2 — IEEE-CIS on authorisation-time fields ({e['version']})",
        "",
        f"{e['dataset']['rows']:,} transactions, {sp['strategy']}; field contract of "
        f"{len(e['protocol']['field_contract'])} raw fields; {len(e['protocol']['velocity_features'])} velocity features.",
        "",
        "| Variant | Val PR-AUC | Test PR-AUC [95% CI] | Δ vs raw [95% CI] | Test ROC-AUC | Recall @ FPR 1% |",
        "|---|---:|---|---|---:|---:|",
    ]
    for k in ("raw_contract", "raw_plus_velocity"):
        c = ci[k]
        delta = (
            "—"
            if "delta_ci_low" not in c
            else (f"{c['delta_vs_raw_contract']:+.3f} [{c['delta_ci_low']:+.3f}, {c['delta_ci_high']:+.3f}]")
        )
        star = " **(selected)**" if k == e["selected_model"] else ""
        lines.append(
            f"| {k.replace('_', ' ')}{star} | {e['validation'][k]['pr_auc']:.3f} | {t[k]['pr_auc']:.3f} "
            f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}] | {delta} | {t[k]['roc_auc']:.3f} | "
            f"{t[k]['recall@fpr=0.01']:.1%} |"
        )
    if e.get("reference_all_431_columns_test_pr_auc") is not None:
        lines += [
            "",
            f"Reference: all 431 columns (incl. vendor C/D/V/id_ aggregates), test PR-AUC "
            f"{e['reference_all_431_columns_test_pr_auc']:.3f}. Share of the raw-to-vendor gap recovered by "
            f"velocity features: {e['share_of_gap_to_vendor_features_recovered']:.0%}.",
        ]
    lines += ["", "Selection: " + "; ".join(e["protocol"]["selection_log"]), ""]
    op = e["selected_operating_points"]["review_threshold_test"]
    lines += [
        f"Operating point (review threshold {e['selected_operating_points']['review_threshold']:.3f}, chosen on "
        f"validation): precision {op['precision']:.1%}, recall {op['recall']:.1%}, {op['alerts']:,} alerts "
        f"({op['alert_rate']:.2%} of transactions).",
        "",
    ]
    for a in e["protocol"].get("amendments", []):
        lines += [f"Protocol amendment: {a}", ""]
    promo = e.get("promotion")
    if promo is not None:
        verdict = "promoted" if promo["promoted"] else "NOT promoted"
        checks = ", ".join(
            f"{c['name']} {'pass' if c['passed'] else 'FAIL'}" for c in promo["report"]["checks"]
        )
        lines += [f"Promotion gate: {verdict} ({checks}).", ""]
    lat = e["latency_online"]
    lines += [
        f"Online latency per transaction (feature-store lookup + update + model): p50 {lat['single_row_p50_ms']:.2f} ms, "
        f"p95 {lat['single_row_p95_ms']:.2f} ms (feature store alone p95 {lat['feature_store_p95_ms']:.2f} ms).",
        "",
    ]
    lines += [
        "## Monthly backtest (label-maturity gap: one month)",
        "",
        "| Scored month | Frauds | Static model PR-AUC | Retrained monthly PR-AUC |",
        "|---:|---:|---:|---:|",
    ]
    for r in e["backtest"]["blocks"]:
        lines.append(
            f"| {r['test_block']} | {r['frauds']} | {r['static_pr_auc']:.3f} | {r['retrained_pr_auc']:.3f} |"
        )
    return "\n".join(lines) + "\n"
