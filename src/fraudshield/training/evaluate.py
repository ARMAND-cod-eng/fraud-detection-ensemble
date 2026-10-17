"""Evaluation utilities for heavily imbalanced fraud scoring.

Conventions
* ``score`` arrays are "higher = more suspicious" (probabilities or anomaly scores).
* Ranking metrics (ROC-AUC, PR-AUC, recall@FPR, precision@budget) are threshold-free.
* Thresholded metrics always take an explicit threshold; thresholds are chosen on validation
  data by the caller, never here.
* Bootstrap CIs resample transactions with replacement (percentile method). With only tens
  of positives in a split, intervals are wide; they are reported rather than hidden.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FPR_TARGETS = (0.001, 0.005, 0.01)
BUDGET_FRACTIONS = (0.001, 0.0025, 0.005)
CALIBRATION_EDGES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def recall_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(y)))
    top = np.argpartition(-score, k - 1)[:k]
    return float(y[top].mean())


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }
    for f in FPR_TARGETS:
        out[f"recall@fpr={f:g}"] = recall_at_fpr(y, score, f)
    for frac in BUDGET_FRACTIONS:
        k = round(len(y) * frac)
        out[f"precision@top{frac:.2%}(k={k})"] = precision_at_k(y, score, k)
    return out


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "alerts": tp + fp,
        "alert_rate": (tp + fp) / len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 (call on validation data only)."""
    order = np.argsort(-score)
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    precision = tp / (tp + fp)
    recall = tp / max(1, yy.sum())
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    # only consider positions where the score changes (a threshold can't split ties)
    valid = np.r_[s[1:] != s[:-1], True]
    i = int(np.argmax(np.where(valid, f1, -1)))
    return float(s[i])


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    bins = []
    ece = 0.0
    edges = np.asarray(CALIBRATION_EDGES)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mean_p, obs = float(p[m].mean()), float(y[m].mean())
        ece += n / len(y) * abs(mean_p - obs)
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "mean_predicted": mean_p,
                "observed_rate": obs,
                "positives": int(y[m].sum()),
            }
        )
    return {"brier": brier, "log_loss": logloss, "ece": float(ece), "reliability": bins}


def downsample_curve(x: np.ndarray, y: np.ndarray, n: int = 400) -> list[list[float]]:
    if len(x) <= n:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.r_[np.linspace(0, len(x) - 1, n).astype(int), len(x) - 1])
    return [[round(float(x[i]), 6), round(float(y[i]), 6)] for i in idx]


def curves(y: np.ndarray, score: np.ndarray) -> dict[str, list[list[float]]]:
    """Compact but exact curves for plotting.

    ROC keeps only corner points (``drop_intermediate``). PR keeps the points where recall
    changes plus their neighbours, i.e. the exact sawtooth at every recovered fraud; index-based
    downsampling would put almost every point at high recall and draw a misleading straight line.
    """
    from sklearn.metrics import precision_recall_curve

    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=True)
    prec, rec, _ = precision_recall_curve(y, score)
    prec, rec = prec[::-1], rec[::-1]  # recall ascending
    change = np.r_[True, rec[1:] != rec[:-1]]
    keep = change | np.r_[change[1:], True]
    return {"roc": downsample_curve(fpr, tpr, 600), "pr": downsample_curve(rec[keep], prec[keep], 600)}


def bootstrap(
    y: np.ndarray,
    scores: Mapping[str, np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    reference: str | None = None,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs per model, plus paired deltas vs ``reference`` (same resamples)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb = y[i]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        for k, s in scores.items():
            samples[k].append(metric(yb, s[i]))
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for k, vals in samples.items():
        v = np.asarray(vals)
        out[k] = {
            "point": float(metric(y, scores[k])),
            "ci_low": float(np.percentile(v, lo_q)),
            "ci_high": float(np.percentile(v, hi_q)),
            "n_boot": len(v),
        }
        if reference is not None and k != reference:
            d = v - np.asarray(samples[reference])
            out[k]["delta_vs_" + reference] = float(metric(y, scores[k]) - metric(y, scores[reference]))
            out[k]["delta_ci_low"] = float(np.percentile(d, lo_q))
            out[k]["delta_ci_high"] = float(np.percentile(d, hi_q))
            out[k]["p_delta_le_0"] = float((d <= 0).mean())
    return out


# --------------------------------------------------------------------------- cost analysis
def review_cost_curve(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    review_cost: float,
    loss_rate: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Expected operational cost of "review if p >= t" under explicit, illustrative assumptions.

    cost(t) = sum(missed fraud amount) * loss_rate + (#reviews) * review_cost
    Assumes reviewed fraud is always caught and reviewing a legitimate transaction only costs
    analyst time. No customer-friction cost because review does not decline the transaction.
    """
    if thresholds is None:
        thresholds = np.unique(np.r_[np.geomspace(1e-4, 0.99, 120), 0.5])
    rows = []
    fraud = y == 1
    for t in thresholds:
        flag = p >= t
        missed = float(amount[fraud & ~flag].sum()) * loss_rate
        reviews = int(flag.sum())
        rows.append(
            {
                "threshold": float(t),
                "reviews": reviews,
                "frauds_caught": int((flag & fraud).sum()),
                "fraud_loss": missed,
                "review_cost": reviews * review_cost,
                "total_cost": missed + reviews * review_cost,
            }
        )
    return rows


def expected_loss_policy_cost(
    y: np.ndarray, p: np.ndarray, amount: np.ndarray, review_cost: float, loss_rate: float = 1.0
) -> dict[str, Any]:
    """Amount-aware rule: review iff p * amount * loss_rate >= review_cost (needs calibrated p)."""
    flag = p * amount * loss_rate >= review_cost
    fraud = y == 1
    missed = float(amount[fraud & ~flag].sum()) * loss_rate
    reviews = int(flag.sum())
    return {
        "rule": "review if p * amount >= review_cost",
        "reviews": reviews,
        "frauds_caught": int((flag & fraud).sum()),
        "fraud_loss": missed,
        "review_cost": reviews * review_cost,
        "total_cost": missed + reviews * review_cost,
    }
