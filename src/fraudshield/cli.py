"""Command-line interface: ``fraudshield <command>``.

fraudshield download-data                 fetch + verify the credit-card dataset
fraudshield train [--promote]             train, select, evaluate, save an artifact
fraudshield evaluate --version V          re-score the test split with a saved artifact
fraudshield models list|show|promote|rollback
fraudshield drift --csv FILE              offline drift report vs. the production reference
fraudshield serve                         run the API (and dashboard if built)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_download(args: argparse.Namespace) -> int:
    from fraudshield.data.download import ensure_creditcard

    path = ensure_creditcard(args.dest, allow_public_mirror=not args.kaggle_only)
    print(f"dataset ready and verified: {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from fraudshield.training.pipeline import load_config, run_training

    cfg = load_config(args.config)
    if args.split:
        cfg["split"]["strategy"] = args.split
    if args.experiment:
        cfg["experiment_name"] = args.experiment
    results = args.results or f"benchmarks/results/{cfg['experiment_name']}"
    out = run_training(
        cfg,
        artifacts_root=args.artifacts,
        results_dir=results,
        data_path=args.data,
        promote=args.promote,
        quick=args.quick,
        make_figures=not args.no_figures,
    )
    ev = out["evaluation"]
    sel = ev["selected_model"]
    _print(
        {
            "version": out["version"],
            "selected_model": sel,
            "validation": ev["validation"][sel],
            "test": {k: v for k, v in ev["test"][sel].items() if k != "at_val_f1_threshold"},
            "review_threshold": ev["selected_operating_points"]["review_threshold"],
        }
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Reproducibility check: reload the artifact, rebuild the split, re-score test."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from fraudshield.data.creditcard import chronological_split, load_creditcard
    from fraudshield.models.artifact import ModelRegistry, load_model

    reg = ModelRegistry(args.artifacts)
    version = args.version or reg.production()
    if version is None:
        print("no version given and no production model", file=sys.stderr)
        return 2
    model, meta = load_model(reg.model_dir(version))
    df, report = load_creditcard(args.data)
    if report["sha256"] != meta["data_sha256"]:
        print("WARNING: data file differs from the one the model was trained on", file=sys.stderr)
    sc = meta["split_config"]
    if sc["strategy"] != "chronological":
        print("evaluate currently supports chronological splits only", file=sys.stderr)
        return 2
    split = chronological_split(df, sc["train_frac"], sc["val_frac"])
    y = split.test["Class"].to_numpy()
    p = model.predict_proba(split.test)
    got = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
    stored = {k: meta["metrics"]["test"][k] for k in got}
    match = all(abs(got[k] - stored[k]) < 1e-3 for k in got)
    _print(
        {
            "version": version,
            "recomputed": got,
            "stored": stored,
            "reproduced": match,
            "note": "PR-AUC is invariant to monotone calibration, so calibrated and raw scores give the same ranking metrics",
        }
    )
    return 0 if match else 1


def cmd_models(args: argparse.Namespace) -> int:
    from fraudshield.models.artifact import ModelRegistry, read_json
    from fraudshield.training.pipeline import load_config
    from fraudshield.training.promotion import PromotionRejected, promote_version

    reg = ModelRegistry(args.artifacts)
    if args.action == "list":
        prod = reg.production()
        rows = []
        for v in reg.versions():
            m = read_json(reg.model_dir(v) / "metadata.json")
            rows.append(
                {
                    "version": v,
                    "production": v == prod,
                    "kind": m["kind"],
                    "test_pr_auc": round(m["metrics"]["test"]["pr_auc"], 4),
                    "test_roc_auc": round(m["metrics"]["test"]["roc_auc"], 4),
                }
            )
        _print({"production": prod, "models": rows})
    elif args.action == "show":
        version = args.version or reg.production()
        if version is None:
            print("no version given and no production model", file=sys.stderr)
            return 2
        _print(read_json(reg.model_dir(version) / "metadata.json"))
    elif args.action == "promote":
        gate = load_config(args.config)["gate"]
        try:
            report = promote_version(reg, args.version, gate, reason=args.reason, force=args.force)
        except PromotionRejected as exc:
            _print(exc.report)
            print("promotion rejected by gate (use --force only after manual review)", file=sys.stderr)
            return 1
        _print(report)
    elif args.action == "rollback":
        print(f"production is now {reg.rollback(args.reason)}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    import pandas as pd

    from fraudshield.inference.scorer import FraudScorer
    from fraudshield.monitoring.drift import drift_report

    scorer = FraudScorer.from_registry(args.artifacts, args.version)
    df = pd.read_csv(args.csv)
    cols = {c: df[c].to_numpy(dtype=np.float64) for c in df.columns}
    s = scorer.score_columns(cols)
    assert scorer.reference is not None
    _print(drift_report(scorer.reference, s.base, scorer.model.feature_builder.feature_names, s.probability))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("api.main:app", host=args.host, port=args.port, workers=1, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fraudshield", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download-data", help="download and verify the credit-card dataset")
    d.add_argument("--dest", default="data/raw/creditcard.csv")
    d.add_argument("--kaggle-only", action="store_true", help="do not fall back to the public mirror")
    d.set_defaults(fn=cmd_download)

    t = sub.add_parser("train", help="train + evaluate + save an artifact")
    t.add_argument("--config", default="configs/training.yaml")
    t.add_argument("--data", default=None)
    t.add_argument("--artifacts", default="artifacts")
    t.add_argument(
        "--results", default=None, help="report directory (default benchmarks/results/<experiment>)"
    )
    t.add_argument("--split", choices=["chronological", "stratified_random"], default=None)
    t.add_argument("--experiment", default=None)
    t.add_argument("--promote", action="store_true", help="promote if the evaluation gate passes")
    t.add_argument("--quick", action="store_true", help="tiny settings (tests / CI)")
    t.add_argument("--no-figures", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="re-score the test split with a saved artifact")
    e.add_argument("--version", default=None)
    e.add_argument("--data", default="data/raw/creditcard.csv")
    e.add_argument("--artifacts", default="artifacts")
    e.set_defaults(fn=cmd_evaluate)

    m = sub.add_parser("models", help="registry: list / show / promote / rollback")
    m.add_argument("action", choices=["list", "show", "promote", "rollback"])
    m.add_argument("--version", default=None)
    m.add_argument("--artifacts", default="artifacts")
    m.add_argument("--config", default="configs/training.yaml")
    m.add_argument("--reason", default="manual")
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_models)

    r = sub.add_parser("drift", help="offline drift report for a CSV of raw transactions")
    r.add_argument("--csv", required=True)
    r.add_argument("--version", default=None)
    r.add_argument("--artifacts", default="artifacts")
    r.set_defaults(fn=cmd_drift)

    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
