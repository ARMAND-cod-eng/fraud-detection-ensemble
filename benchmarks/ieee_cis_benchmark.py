"""Second-dataset benchmark: IEEE-CIS, with and without past-only entity features.

Variants (same XGBoost settings, out-of-time 60/20/20 split on TransactionDT):
* ``none``        - original columns only
* ``cumulative``  - + entity counts since the start of the data (grow with time)
* ``windowed``    - + entity counts over trailing 24 h / 7 day windows

Pre-registered rule: the variant with the best VALIDATION PR-AUC is "selected"; test results are
reported for every variant with paired-bootstrap differences vs ``none``. Results go to
``benchmarks/results/ieee-cis/`` and are never merged with the credit-card benchmark.

Prerequisite: accept the competition rules on Kaggle and extract the training files (see
``src/fraudshield/data/ieee_cis.py``).

    python benchmarks/ieee_cis_benchmark.py --root data/raw/ieee-cis
    python benchmarks/ieee_cis_benchmark.py --synthetic   # pipeline smoke test only; numbers meaningless
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudshield.data.ieee_cis import IEEEPrepared, load_ieee_cis, prepare, synthetic_ieee_like
from fraudshield.training.evaluate import bootstrap, ranking_metrics


def fit_eval(p: IEEEPrepared, mode: str | None, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    cols = p.columns(mode)
    clf = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(p.x_train[cols], p.y_train, eval_set=[(p.x_val[cols], p.y_val)], verbose=False)
    pv = clf.predict_proba(p.x_val[cols])[:, 1]
    pt = clf.predict_proba(p.x_test[cols])[:, 1]
    gain = clf.get_booster().get_score(importance_type="gain")
    entity = p.entity_features.get(mode or "", [])
    total = sum(gain.values()) or 1.0
    return {
        "features": len(cols),
        "trees": int(clf.best_iteration) + 1,
        "validation": {
            "pr_auc": float(average_precision_score(p.y_val, pv)),
            "roc_auc": float(roc_auc_score(p.y_val, pv)),
        },
        "test": ranking_metrics(p.y_test, pt),
        "entity_feature_gain_share": float(sum(gain.get(f, 0.0) for f in entity) / total),
    }, pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw/ieee-cis")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["none", "cumulative", "windowed"],
        choices=["none", "cumulative", "windowed"],
    )
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default="benchmarks/results/ieee-cis/metrics.json")
    args = ap.parse_args()
    if "none" not in args.variants:
        args.variants.insert(0, "none")

    t0 = time.perf_counter()
    df = synthetic_ieee_like() if args.synthetic else load_ieee_cis(args.root, nrows=args.nrows)
    rows, fraud_rate = len(df), float(df["isFraud"].mean())
    gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"loaded {df.shape} in {time.perf_counter() - t0:.0f}s, {gb:.2f} GB", flush=True)
    modes = tuple(v for v in args.variants if v != "none")
    p = prepare(df, graph=bool(modes), modes=modes)
    del df
    print(
        f"prepared in {time.perf_counter() - t0:.0f}s: train {len(p.y_train)} ({p.y_train.mean():.4f} fraud), "
        f"val {len(p.y_val)}, test {len(p.y_test)} ({p.y_test.mean():.4f}); "
        + ", ".join(f"{m}: {len(f)} entity features" for m, f in p.entity_features.items()),
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    scores: dict[str, np.ndarray] = {}
    for v in args.variants:
        results[v], scores[v] = fit_eval(p, None if v == "none" else v, seed=42)
        r = results[v]
        print(
            f"{v:10s}: val PR-AUC {r['validation']['pr_auc']:.4f}  test PR-AUC {r['test']['pr_auc']:.4f}  "
            f"trees {r['trees']}  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    selected = max(results, key=lambda v: results[v]["validation"]["pr_auc"])
    ci = bootstrap(p.y_test, scores, average_precision_score, n_boot=args.bootstrap, reference="none")
    out = {
        "dataset": "SYNTHETIC smoke test (not real data)"
        if args.synthetic
        else "IEEE-CIS train_transaction + train_identity",
        "rows": rows,
        "split": "chronological 60/20/20 on TransactionDT",
        "fraud_rate": fraud_rate,
        "test_fraud_rate": float(p.y_test.mean()),
        "selection_rule": "best validation PR-AUC among variants (test reported for all)",
        "selected_variant": selected,
        "entity_features": p.entity_features,
        "variants": results,
        "test_pr_auc_95ci": ci,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    summary = {
        v: {"val": round(r["validation"]["pr_auc"], 4), "test": round(r["test"]["pr_auc"], 4)}
        for v, r in results.items()
    }
    deltas = {
        v: [round(c["delta_ci_low"], 4), round(c["delta_ci_high"], 4)]
        for v, c in ci.items()
        if "delta_ci_low" in c
    }
    print(json.dumps({"selected": selected, "pr_auc": summary, "delta_vs_none_95ci": deltas}, indent=2))


if __name__ == "__main__":
    main()
