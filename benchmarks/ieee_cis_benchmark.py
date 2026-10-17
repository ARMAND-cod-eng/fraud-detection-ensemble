"""Second-dataset benchmark: IEEE-CIS, with and without past-only entity/graph features.

Prerequisite: accept the competition rules on Kaggle and download the data (see
``src/fraudshield/data/ieee_cis.py``). Results go to ``benchmarks/results/ieee-cis/`` and are
never merged with the credit-card benchmark.

    python benchmarks/ieee_cis_benchmark.py --root data/raw/ieee-cis
    python benchmarks/ieee_cis_benchmark.py --synthetic   # pipeline smoke test only; numbers meaningless
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudshield.data.ieee_cis import load_ieee_cis, prepare, synthetic_ieee_like
from fraudshield.training.evaluate import bootstrap, ranking_metrics


def fit_eval(p, use_graph: bool, seed: int) -> tuple[dict, np.ndarray]:
    cols = [c for c in p.feature_names if use_graph or c not in p.graph_features]
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
    return {
        "features": len(cols),
        "trees": int(clf.best_iteration) + 1,
        "validation": {
            "pr_auc": float(average_precision_score(p.y_val, pv)),
            "roc_auc": float(roc_auc_score(p.y_val, pv)),
        },
        "test": ranking_metrics(p.y_test, pt),
    }, pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw/ieee-cis")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default="benchmarks/results/ieee-cis/metrics.json")
    args = ap.parse_args()

    df = synthetic_ieee_like() if args.synthetic else load_ieee_cis(args.root, nrows=args.nrows)
    p = prepare(df, graph=True)
    base, s_base = fit_eval(p, use_graph=False, seed=42)
    with_graph, s_graph = fit_eval(p, use_graph=True, seed=42)
    ci = bootstrap(
        p.y_test,
        {"no_graph": s_base, "graph": s_graph},
        average_precision_score,
        n_boot=args.bootstrap,
        reference="no_graph",
    )
    out = {
        "dataset": "SYNTHETIC smoke test (not real data)"
        if args.synthetic
        else "IEEE-CIS train_transaction + train_identity",
        "rows": len(df),
        "split": "chronological 60/20/20 on TransactionDT",
        "fraud_rate": float(df["isFraud"].mean()),
        "graph_features": p.graph_features,
        "xgboost_without_graph_features": base,
        "xgboost_with_graph_features": with_graph,
        "test_pr_auc_95ci": ci,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "pr_auc_no_graph": base["test"]["pr_auc"],
                "pr_auc_graph": with_graph["test"]["pr_auc"],
                "delta_ci": [ci["graph"]["delta_ci_low"], ci["graph"]["delta_ci_high"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
