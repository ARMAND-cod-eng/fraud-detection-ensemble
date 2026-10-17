"""Reproducible training, model selection and evaluation.

Protocol (see docs/ML_EVALUATION.md):
1. Load + validate data, drop exact duplicates, out-of-time split 60/20/20.
2. Fit every candidate on TRAIN. XGBoost early stopping uses a slice of TRAIN.
3. Hyper-parameters, model choice, calibration map and thresholds use VALIDATION only.
4. TEST is scored once, after every decision above is frozen, for the final report.
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudshield import __version__
from fraudshield.data.creditcard import Split, chronological_split, load_creditcard, stratified_random_split
from fraudshield.data.schema import AMOUNT_COLUMN, LABEL_COLUMN, RAW_FEATURE_COLUMNS, TIME_COLUMN
from fraudshield.features.engineering import FeatureBuilder
from fraudshield.models.anomaly import IsolationForestDetector
from fraudshield.models.artifact import ModelRegistry, save_model
from fraudshield.models.autoencoder import AutoencoderConfig, AutoencoderDetector
from fraudshield.models.calibration import PlattCalibrator
from fraudshield.models.fraud_model import SIGNAL_NAMES, FraudModel, LinearModel
from fraudshield.monitoring.drift import build_reference
from fraudshield.training import evaluate as ev


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- model fitting
def fit_xgb(
    x: np.ndarray, y: np.ndarray, params: dict[str, Any], es_frac: float, seed: int
) -> tuple[xgb.Booster, int]:
    p = dict(params)
    n_estimators = int(p.pop("n_estimators"))
    es_rounds = int(p.pop("early_stopping_rounds"))
    idx_fit, idx_es = train_test_split(np.arange(len(y)), test_size=es_frac, random_state=seed, stratify=y)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, early_stopping_rounds=es_rounds, random_state=seed, n_jobs=-1, **p
    )
    clf.fit(x[idx_fit], y[idx_fit], eval_set=[(x[idx_es], y[idx_es])], verbose=False)
    best = int(clf.best_iteration)
    booster = clf.get_booster()[: best + 1]  # keep only the trees selected by early stopping
    return booster, best


def booster_proba(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    return np.asarray(booster.inplace_predict(x), dtype=np.float64)


def fit_logreg(x: np.ndarray, y: np.ndarray, c: float, class_weight: str | None, seed: int) -> LinearModel:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight=class_weight, max_iter=2000, random_state=seed),
    )
    pipe.fit(x, y)
    return LinearModel.from_sklearn(pipe)


def linear_proba(lm: LinearModel, x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-lm.margin(x)))


def fit_detectors(
    x: np.ndarray, y: np.ndarray, cfg: dict[str, Any], seed: int
) -> tuple[IsolationForestDetector, AutoencoderDetector]:
    """Isolation Forest on all rows (no labels); autoencoder on normal rows only."""
    iso = IsolationForestDetector(seed=seed, **cfg["isolation_forest"]).fit(x)
    ae_cfg = dict(cfg["autoencoder"])
    ae_cfg["hidden"] = tuple(ae_cfg["hidden"])
    ae = AutoencoderDetector(AutoencoderConfig(seed=seed, **ae_cfg)).fit(x[y == 0])
    return iso, ae


def detector_signals(iso: IsolationForestDetector, ae: AutoencoderDetector, x: np.ndarray) -> np.ndarray:
    return np.column_stack([iso.score(x), ae.score(x)])


def oof_signals(x: np.ndarray, y: np.ndarray, cfg: dict[str, Any], seed: int) -> np.ndarray:
    """Out-of-fold detector scores for training rows.

    The legacy notebook trained the autoencoder on training normals and then fed its in-sample
    reconstruction errors to XGBoost: training normals look 'more normal' than unseen normals,
    a label-dependent artefact the classifier can exploit. Out-of-fold scores remove it.
    """
    out = np.zeros((len(y), 2))
    folds = StratifiedKFold(n_splits=int(cfg["stacking"]["oof_folds"]), shuffle=True, random_state=seed)
    for k, (fit_idx, hold_idx) in enumerate(folds.split(x, y)):
        iso, ae = fit_detectors(x[fit_idx], y[fit_idx], cfg, seed + k + 1)
        out[hold_idx] = detector_signals(iso, ae, x[hold_idx])
    return out


# --------------------------------------------------------------------------- latency
def measure_latency(model: FraudModel, raw: pd.DataFrame, repeats: int) -> dict[str, float]:
    model.prepare_for_serving()
    cols = list(RAW_FEATURE_COLUMNS)
    arr = raw[cols].to_numpy(dtype=np.float64)
    times = []
    for i in range(repeats + 20):
        rec = {c: arr[i % len(arr), j : j + 1] for j, c in enumerate(cols)}  # API-like dict input
        t = time.perf_counter()
        model.predict_from_inputs(model.model_inputs(model.base_features(rec)))
        if i >= 20:  # discard warm-up iterations
            times.append((time.perf_counter() - t) * 1000)
    a = np.asarray(times)
    n_batch = min(10_000, len(raw))
    t = time.perf_counter()
    model.predict_proba(raw.iloc[:n_batch])
    batch_s = time.perf_counter() - t
    return {
        "single_row_p50_ms": float(np.percentile(a, 50)),
        "single_row_p95_ms": float(np.percentile(a, 95)),
        "single_row_p99_ms": float(np.percentile(a, 99)),
        "single_row_mean_ms": float(a.mean()),
        "batch_rows": n_batch,
        "batch_rows_per_s": float(n_batch / batch_s),
        "repeats": repeats,
    }


# --------------------------------------------------------------------------- pipeline
@dataclass
class Candidate:
    name: str
    family: str  # supervised | unsupervised
    val_score: np.ndarray
    test_score: np.ndarray
    model: FraudModel | None = None
    notes: str = ""
    info: dict[str, Any] = field(default_factory=dict)


def _make_split(df: pd.DataFrame, cfg: dict[str, Any]) -> Split:
    s = cfg["split"]
    if s["strategy"] == "chronological":
        return chronological_split(df, s["train_frac"], s["val_frac"])
    if s["strategy"] == "stratified_random":
        return stratified_random_split(df, s["train_frac"], s["val_frac"], seed=cfg["seed"])
    raise ValueError(f"unknown split strategy {s['strategy']}")


def run_training(
    config: dict[str, Any],
    artifacts_root: str | Path = "artifacts",
    results_dir: str | Path | None = None,
    data_path: str | Path | None = None,
    promote: bool = False,
    export_replay: bool = True,
    make_figures: bool = True,
    quick: bool = False,
) -> dict[str, Any]:
    """Train all candidates, select on validation, evaluate once on test, save the artifact."""
    t_start = time.perf_counter()
    cfg = json.loads(json.dumps(config))  # deep copy
    seed = int(cfg["seed"])
    np.random.seed(seed)
    if quick:  # tiny settings for tests/CI on synthetic data
        cfg["xgboost"]["grid"] = {"max_depth": [3], "min_child_weight": [1], "scale_pos_weight": [1]}
        cfg["xgboost"]["base_params"]["n_estimators"] = 200
        cfg["xgboost"]["base_params"]["early_stopping_rounds"] = 20
        cfg["logreg"]["grid"] = {"C": [0.1], "class_weight": [None]}
        cfg["autoencoder"]["epochs"] = 3
        cfg["isolation_forest"]["n_estimators"] = 50
        cfg["stacking"]["oof_folds"] = 2
        cfg["evaluation"]["bootstrap"] = 50
        cfg["evaluation"]["latency_repeats"] = 50

    data_file = Path(data_path or cfg["data"]["path"])
    df, data_report = load_creditcard(data_file, drop_duplicates=cfg["data"]["drop_duplicates"])
    split = _make_split(df, cfg)
    _log(f"data: {data_report['rows']} rows, split {split.summary()}")

    fb = FeatureBuilder(engineered=cfg["features"]["engineered"]).fit(split.train)
    x_tr, x_va, x_te = (fb.transform(p) for p in (split.train, split.validation, split.test))
    y_tr, y_va, y_te = (p[LABEL_COLUMN].to_numpy() for p in (split.train, split.validation, split.test))
    amt_va, amt_te = split.validation[AMOUNT_COLUMN].to_numpy(), split.test[AMOUNT_COLUMN].to_numpy()
    es_frac = float(cfg["xgboost"]["early_stopping_frac"])
    candidates: dict[str, Candidate] = {}

    # ---- XGBoost grid (validation PR-AUC) ------------------------------------------------
    grid_cfg = cfg["xgboost"]["grid"]
    keys = list(grid_cfg)
    xgb_grid = []
    best_xgb: tuple[float, dict[str, Any], xgb.Booster, int] | None = None
    for values in itertools.product(*(grid_cfg[k] for k in keys)):
        params = {**cfg["xgboost"]["base_params"], **dict(zip(keys, values, strict=True))}
        booster, best_it = fit_xgb(x_tr, y_tr, params, es_frac, seed)
        ap = float(average_precision_score(y_va, booster_proba(booster, x_va)))
        xgb_grid.append(
            {"params": dict(zip(keys, values, strict=True)), "best_iteration": best_it, "val_pr_auc": ap}
        )
        _log(f"xgboost {dict(zip(keys, values, strict=True))} trees={best_it + 1} val PR-AUC={ap:.4f}")
        if best_xgb is None or ap > best_xgb[0]:
            best_xgb = (ap, params, booster, best_it)
    assert best_xgb is not None
    _, xgb_params, xgb_booster, xgb_best_it = best_xgb
    xgb_model = FraudModel("xgboost", fb, booster=xgb_booster)
    candidates["xgboost"] = Candidate(
        "xgboost",
        "supervised",
        booster_proba(xgb_booster, x_va),
        booster_proba(xgb_booster, x_te),
        xgb_model,
        info={
            "params": {
                k: v for k, v in xgb_params.items() if k not in ("n_estimators", "early_stopping_rounds")
            },
            "trees": xgb_best_it + 1,
            "grid": xgb_grid,
        },
    )

    # ---- Logistic regression baseline ----------------------------------------------------
    lr_grid, best_lr = [], None
    for c, cw in itertools.product(cfg["logreg"]["grid"]["C"], cfg["logreg"]["grid"]["class_weight"]):
        lm = fit_logreg(x_tr, y_tr, c, cw, seed)
        ap = float(average_precision_score(y_va, linear_proba(lm, x_va)))
        lr_grid.append({"C": c, "class_weight": cw, "val_pr_auc": ap})
        if best_lr is None or ap > best_lr[0]:
            best_lr = (ap, lm, {"C": c, "class_weight": cw})
    assert best_lr is not None
    lr_model = FraudModel("logreg", fb, linear=best_lr[1])
    candidates["logreg"] = Candidate(
        "logreg",
        "supervised",
        linear_proba(best_lr[1], x_va),
        linear_proba(best_lr[1], x_te),
        lr_model,
        info={"params": best_lr[2], "grid": lr_grid},
    )
    _log(f"logreg best {best_lr[2]} val PR-AUC={best_lr[0]:.4f}")

    # ---- Feature ablation: XGBoost on raw columns only -----------------------------------
    fb_raw = FeatureBuilder(engineered=False).fit(split.train)
    xr_tr, xr_va, xr_te = (fb_raw.transform(p) for p in (split.train, split.validation, split.test))
    raw_booster, raw_it = fit_xgb(xr_tr, y_tr, xgb_params, es_frac, seed)
    candidates["xgboost_raw_features"] = Candidate(
        "xgboost_raw_features",
        "supervised",
        booster_proba(raw_booster, xr_va),
        booster_proba(raw_booster, xr_te),
        notes="ablation: V1-V28 + Amount only, same hyper-parameters",
        info={"trees": raw_it + 1},
    )

    # ---- Unsupervised detectors ----------------------------------------------------------
    _log("fitting isolation forest + autoencoder on train")
    iso, ae = fit_detectors(x_tr, y_tr, cfg, seed)
    sig_tr_in, sig_va, sig_te = (detector_signals(iso, ae, x) for x in (x_tr, x_va, x_te))
    candidates["isolation_forest"] = Candidate(
        "isolation_forest", "unsupervised", sig_va[:, 0], sig_te[:, 0], notes="label-free anomaly score"
    )
    candidates["autoencoder"] = Candidate(
        "autoencoder",
        "unsupervised",
        sig_va[:, 1],
        sig_te[:, 1],
        notes="trained on normal transactions only",
        info={
            "epochs_run": len(ae.history_["train_loss"]),
            "holdout_loss_first_last": [ae.history_["holdout_loss"][0], ae.history_["holdout_loss"][-1]],
        },
    )

    # ---- Stacked ensembles ---------------------------------------------------------------
    _log("computing out-of-fold detector scores for stacking")
    sig_tr_oof = oof_signals(x_tr, y_tr, cfg, seed)
    for name, sig_train, note in (
        ("stacked", sig_tr_oof, "XGBoost + out-of-fold Isolation Forest/autoencoder scores"),
        ("stacked_insample", sig_tr_in, "legacy design: in-sample detector scores (leaky; reference only)"),
    ):
        booster, it = fit_xgb(np.hstack([x_tr, sig_train]), y_tr, xgb_params, es_frac, seed)
        m = FraudModel("stacked", fb, booster=booster, iforest=iso, autoencoder=ae)
        candidates[name] = Candidate(
            name,
            "supervised",
            booster_proba(booster, np.hstack([x_va, sig_va])),
            booster_proba(booster, np.hstack([x_te, sig_te])),
            m if name == "stacked" else None,
            notes=note,
            info={"trees": it + 1},
        )
        _log(
            f"{name}: trees={it + 1} val PR-AUC={average_precision_score(y_va, candidates[name].val_score):.4f}"
        )

    # ---- Validation metrics + pre-registered selection rule ------------------------------
    val_metrics = {k: ev.ranking_metrics(y_va, c.val_score) for k, c in candidates.items()}
    sel = cfg["selection"]
    simpler = sel["simpler_model"]
    val_boot = ev.bootstrap(
        y_va,
        {k: candidates[k].val_score for k in sel["eligible"]},
        ev.average_precision_score,
        n_boot=int(cfg["evaluation"]["bootstrap"]),
        seed=seed,
        reference=simpler,
        alpha=0.10,
    )
    selected, reasons = simpler, []
    for k in sel["eligible"]:
        if k == simpler:
            continue
        gain = val_metrics[k]["pr_auc"] - val_metrics[simpler]["pr_auc"]
        ci_lo = val_boot[k].get("delta_ci_low", -1.0)
        ok = gain >= sel["min_improvement"] and ci_lo > 0
        reasons.append(
            f"{k}: val PR-AUC gain {gain:+.4f} vs {simpler} (90% CI low {ci_lo:+.4f}) -> "
            f"{'qualifies' if ok else 'does not qualify'}"
        )
        if ok and val_metrics[k]["pr_auc"] > val_metrics[selected]["pr_auc"]:
            selected = k
    reasons.append(f"selected: {selected}")
    _log(" | ".join(reasons))
    chosen = candidates[selected]
    assert chosen.model is not None
    model = chosen.model

    # ---- Calibration + thresholds (validation) -------------------------------------------
    model.calibrator = PlattCalibrator().fit(chosen.val_score, y_va)
    p_va, p_te = model.calibrator.transform(chosen.val_score), model.calibrator.transform(chosen.test_score)
    costs = cfg["cost_assumptions"]
    val_curve = ev.review_cost_curve(y_va, p_va, amt_va, costs["review_cost"], costs["loss_rate"])
    review_threshold = min(val_curve, key=lambda r: r["total_cost"])["threshold"]
    f1_thresholds = {k: ev.best_f1_threshold(y_va, c.val_score) for k, c in candidates.items()}
    f1_threshold_cal = ev.best_f1_threshold(y_va, p_va)

    # ---- Final test evaluation (test is used from here on, for reporting only) -----------
    _log("scoring test split (final report)")
    test_metrics: dict[str, Any] = {}
    for k, c in candidates.items():
        test_metrics[k] = {
            **ev.ranking_metrics(y_te, c.test_score),
            "at_val_f1_threshold": ev.threshold_metrics(y_te, c.test_score, f1_thresholds[k]),
        }
    n_boot = int(cfg["evaluation"]["bootstrap"])
    scores_te = {k: c.test_score for k, c in candidates.items()}
    ci_pr = ev.bootstrap(
        y_te, scores_te, ev.average_precision_score, n_boot=n_boot, seed=seed, reference="xgboost"
    )
    ci_roc = ev.bootstrap(y_te, scores_te, ev.roc_auc_score, n_boot=n_boot, seed=seed, reference="xgboost")
    op_review = ev.threshold_metrics(y_te, p_te, review_threshold)
    op_f1 = ev.threshold_metrics(y_te, p_te, f1_threshold_cal)
    recall_ci = ev.bootstrap(
        y_te,
        {"selected": p_te},
        lambda yy, ss: ev.threshold_metrics(yy, ss, review_threshold)["recall"],
        n_boot=n_boot,
        seed=seed,
    )["selected"]
    precision_ci = ev.bootstrap(
        y_te,
        {"selected": p_te},
        lambda yy, ss: ev.threshold_metrics(yy, ss, review_threshold)["precision"],
        n_boot=n_boot,
        seed=seed,
    )["selected"]

    test_curve = ev.review_cost_curve(y_te, p_te, amt_te, costs["review_cost"], costs["loss_rate"])

    def at(t: float) -> dict[str, float]:
        return ev.review_cost_curve(
            y_te, p_te, amt_te, costs["review_cost"], costs["loss_rate"], np.array([t])
        )[0]

    cost_experiment = {
        "assumptions": {
            **costs,
            "note": "Illustrative assumptions; no real loss or review-cost data available.",
        },
        "validation_cost_optimal_threshold": review_threshold,
        "test": {
            "approve_all": {"total_cost": float(amt_te[y_te == 1].sum()) * costs["loss_rate"], "reviews": 0},
            "review_all": {"total_cost": len(y_te) * costs["review_cost"], "reviews": len(y_te)},
            "threshold_0.5": at(0.5),
            "validation_cost_optimal": at(review_threshold),
            "expected_loss_rule": ev.expected_loss_policy_cost(
                y_te, p_te, amt_te, costs["review_cost"], costs["loss_rate"]
            ),
            "test_oracle_threshold_for_reference_only": min(test_curve, key=lambda r: r["total_cost"]),
        },
        "validation_curve": val_curve,
        "test_curve": test_curve,
    }

    # ---- Latency, explainability, drift reference ----------------------------------------
    repeats = int(cfg["evaluation"]["latency_repeats"])
    latency = {}
    for k in ("logreg", "xgboost", "stacked"):
        cand_model = candidates[k].model
        assert cand_model is not None
        latency[k] = measure_latency(cand_model, split.test, repeats)
        _log(
            f"latency {k}: p50={latency[k]['single_row_p50_ms']:.3f} ms p95={latency[k]['single_row_p95_ms']:.3f} ms"
        )

    model.signal_quantiles = {
        name: np.quantile(sig_va[y_va == 0, i], np.linspace(0, 1, 101)).tolist()
        for i, name in enumerate(SIGNAL_NAMES)
    }
    if not model.has_detectors:  # keep detectors as auxiliary analyst signals even if not used for scoring
        model.iforest, model.autoencoder = iso, ae
    contrib, _ = model.contributions(model.model_inputs(x_va))
    importance = np.abs(contrib).mean(axis=0)
    order = np.argsort(-importance)
    global_importance = [
        {"feature": model.model_feature_names[i], "mean_abs_shap": float(importance[i])} for i in order
    ]

    reference = build_reference(
        x_tr, fb.feature_names, p_va, alert_rate=float((p_va >= review_threshold).mean())
    )

    # ---- Assemble report ------------------------------------------------------------------
    created = datetime.now(UTC)
    version = f"{selected}-{created:%Y%m%d-%H%M%S}"
    calib = {
        "raw": ev.calibration_metrics(y_te, chosen.test_score),
        "calibrated": ev.calibration_metrics(y_te, p_te),
    }
    evaluation = {
        "version": version,
        "experiment": cfg["experiment_name"],
        "created_at": created.isoformat(timespec="seconds"),
        "dataset": {**data_report, "split": split.summary()},
        "protocol": {
            "selection_metric": "validation PR-AUC",
            "selection_rule": sel,
            "selection_log": reasons,
            "calibration": "Platt scaling fit on validation",
            "thresholds": "chosen on validation (cost-optimal review threshold; F1-optimal for comparisons)",
            "test_usage": "scored once after all choices were frozen",
        },
        "selected_model": selected,
        "candidates": {
            k: {"family": c.family, "notes": c.notes, "info": c.info} for k, c in candidates.items()
        },
        "validation": val_metrics,
        "validation_bootstrap_pr_auc_90ci": val_boot,
        "test": test_metrics,
        "test_bootstrap_95ci": {"pr_auc": ci_pr, "roc_auc": ci_roc},
        "selected_operating_points": {
            "review_threshold": review_threshold,
            "review_threshold_test": op_review,
            "review_threshold_test_recall_ci": recall_ci,
            "review_threshold_test_precision_ci": precision_ci,
            "f1_threshold": f1_threshold_cal,
            "f1_threshold_test": op_f1,
        },
        "calibration_test": calib,
        "cost_experiment": cost_experiment,
        "latency_in_process": latency,
        "global_importance": global_importance[:25],
        "curves_test": {k: ev.curves(y_te, c.test_score) for k, c in candidates.items()},
        "environment": {
            "fraudshield": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "git_commit": _git_commit(),
        },
        "config": cfg,
        "runtime_seconds": None,
    }
    gate_cfg = cfg["gate"]
    gate_checks = {
        "test_pr_auc": [
            test_metrics[selected]["pr_auc"],
            ">=",
            gate_cfg["min_test_pr_auc"],
            test_metrics[selected]["pr_auc"] >= gate_cfg["min_test_pr_auc"],
        ],
        "test_roc_auc": [
            test_metrics[selected]["roc_auc"],
            ">=",
            gate_cfg["min_test_roc_auc"],
            test_metrics[selected]["roc_auc"] >= gate_cfg["min_test_roc_auc"],
        ],
        "p95_latency_ms": [
            latency[selected]["single_row_p95_ms"],
            "<=",
            gate_cfg["max_p95_latency_ms"],
            latency[selected]["single_row_p95_ms"] <= gate_cfg["max_p95_latency_ms"],
        ],
    }
    metadata = {
        "version": version,
        "kind": model.kind,
        "selected_candidate": selected,
        "created_at": evaluation["created_at"],
        "data_sha256": data_report["sha256"],
        "split": split.summary(),
        "split_config": cfg["split"],
        "feature_names": fb.feature_names,
        "model_feature_names": model.model_feature_names,
        "thresholds": {"review": review_threshold, "f1": f1_threshold_cal},
        "metrics": {
            "validation": val_metrics[selected],
            "test": {k: v for k, v in test_metrics[selected].items() if k != "at_val_f1_threshold"},
        },
        "gate_checks": gate_checks,
        "environment": evaluation["environment"],
        "training_config_digest": __import__("hashlib")
        .sha256(json.dumps(cfg, sort_keys=True).encode())
        .hexdigest()[:12],
    }

    registry = ModelRegistry(artifacts_root)
    model_dir = save_model(model, registry.models_dir / version, metadata, reference, evaluation)
    _log(f"saved model artifact {model_dir}")
    if promote:
        from fraudshield.training.promotion import promote_version

        promote_version(registry, version, gate_cfg, reason="trained by pipeline")

    if export_replay:
        replay_dir = Path(artifacts_root) / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        cols = [TIME_COLUMN, *[c for c in RAW_FEATURE_COLUMNS if c != TIME_COLUMN], LABEL_COLUMN]
        split.test[cols].to_csv(replay_dir / "creditcard_test_window.csv.gz", index=False, compression="gzip")

    evaluation["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    if results_dir is not None:
        from fraudshield.training.report import write_report

        write_report(evaluation, Path(results_dir), make_figures=make_figures)
        _log(f"wrote benchmark report to {results_dir}")

    from fraudshield.training.tracking import log_run

    log_run(cfg, evaluation, model_dir, results_dir)
    _log(f"done in {evaluation['runtime_seconds']} s")
    return {"version": version, "model_dir": str(model_dir), "evaluation": evaluation}
