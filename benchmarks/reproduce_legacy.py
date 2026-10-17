"""Faithful reproduction of the original notebook (notebooks/01_fraud_detection_complete.ipynb).

Purpose: verify the historical claims (ROC-AUC 0.9834, recall 85.71%, 86 ms latency)
with the *same* methodology, then quantify the methodological issues found in the audit.

Deviations from the notebook, all forced by library changes:
  * XGBoost >= 2.0 moved ``early_stopping_rounds`` from ``fit()`` to the constructor and
    removed ``use_label_encoder``. Behaviour is otherwise identical.
  * Runs on CPU instead of a Kaggle GPU.

Everything else (feature engineering fit on the full dataset, random stratified split with
seed 42, RobustScaler, autoencoder with Sigmoid output, Isolation Forest, XGBoost settings,
fixed 0.5 threshold, $500 assumed fraud amount) is reproduced as written, *including bugs*,
because the goal is to test the reported numbers, not to improve them.

Usage:
    python benchmarks/reproduce_legacy.py --data data/raw/creditcard.csv \
        --out benchmarks/results/legacy_reproduction.json [--seeds 42 0 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

CLAIMED = {"roc_auc": 0.9834, "recall": 0.8571, "precision": 0.4746, "f1": 0.6109, "latency_ms": 86.0}


# --------------------------------------------------------------------------------------
# Verbatim port of the notebook's FeatureEngineer (cell 15), verbose output removed.
# --------------------------------------------------------------------------------------
class FeatureEngineer:
    def __init__(self) -> None:
        self.fitted_features = None
        self.feature_stats: dict = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        for col in df.columns:
            if col.startswith("V") or col == "Amount":
                self.feature_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                }
        self.is_fitted = True
        return self

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = df.copy()
        is_single_row = len(df_feat) == 1
        if not self.is_fitted and not is_single_row:
            self.fit(df_feat)
        if "Amount" in df_feat.columns:
            df_feat["Amount_log"] = np.log1p(df_feat["Amount"])
            df_feat["Amount_squared"] = df_feat["Amount"] ** 2
            df_feat["Amount_sqrt"] = np.sqrt(np.abs(df_feat["Amount"]))
            if "Amount" in self.feature_stats:
                m, s = self.feature_stats["Amount"]["mean"], self.feature_stats["Amount"]["std"]
                df_feat["Amount_zscore"] = np.abs((df_feat["Amount"] - m) / s) if s > 0 else 0
            else:
                df_feat["Amount_zscore"] = 0
        if "Time" in df_feat.columns:
            df_feat["Hour"] = (df_feat["Time"] / 3600) % 24
            df_feat["Time_sin"] = np.sin(2 * np.pi * df_feat["Hour"] / 24)
            df_feat["Time_cos"] = np.cos(2 * np.pi * df_feat["Hour"] / 24)
        pca_cols = [c for c in df_feat.columns if c.startswith("V")]
        if pca_cols:
            if is_single_row:
                vals = df_feat[pca_cols].values.flatten()
                df_feat["PCA_mean"] = np.mean(vals)
                df_feat["PCA_std"] = np.std(vals) if len(vals) > 1 else 0
                df_feat["PCA_max"] = np.max(vals)
                df_feat["PCA_min"] = np.min(vals)
                df_feat["PCA_range"] = df_feat["PCA_max"] - df_feat["PCA_min"]
            else:
                df_feat["PCA_mean"] = df_feat[pca_cols].mean(axis=1)
                df_feat["PCA_std"] = df_feat[pca_cols].std(axis=1)
                df_feat["PCA_max"] = df_feat[pca_cols].max(axis=1)
                df_feat["PCA_min"] = df_feat[pca_cols].min(axis=1)
                df_feat["PCA_range"] = df_feat["PCA_max"] - df_feat["PCA_min"]
            for i in [1, 2, 3, 4, 14, 17]:
                col = f"V{i}"
                if col in df_feat.columns and col in self.feature_stats:
                    m, s = self.feature_stats[col]["mean"], self.feature_stats[col]["std"]
                    df_feat[f"{col}_zscore"] = np.abs((df_feat[col] - m) / s) if s > 0 else 0
        df_feat = df_feat.replace([np.inf, -np.inf], 0).fillna(0)
        if self.fitted_features is None:
            self.fitted_features = [c for c in df_feat.columns if c not in ["Class", "Time", "Hour"]]
        return df_feat


class FraudAutoencoder(nn.Module):
    """Notebook cell 19, unchanged (note the Sigmoid output on unbounded inputs)."""

    def __init__(self, input_dim: int, encoding_dim: int = 14) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(0.2),
            nn.Linear(24, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(0.2),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return nn.functional.mse_loss(self.forward(x), x, reduction="none").mean(dim=1)


def ensemble_features(x_scaled: np.ndarray, ae: FraudAutoencoder, iso: IsolationForest) -> np.ndarray:
    ae.eval()
    err = ae.get_reconstruction_error(torch.FloatTensor(x_scaled)).numpy()
    return np.column_stack([x_scaled, err.reshape(-1, 1), iso.decision_function(x_scaled).reshape(-1, 1)])


def business_impact(y_true: np.ndarray, proba: np.ndarray, threshold: float, fraud_amount=None) -> dict:
    """Notebook cell 37. ``fraud_amount=None`` reproduces the $500 flat assumption."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    if fraud_amount is None:
        prevented, losses = tp * 500.0, fn * 500.0
    else:
        prevented = float(fraud_amount[(pred == 1) & (y_true == 1)].sum())
        losses = float(fraud_amount[(pred == 0) & (y_true == 1)].sum())
    inv, fric = (tp + fp) * 10.0, fp * 5.0
    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "fraud_prevented": round(prevented, 2),
        "net_benefit": round(prevented - losses - inv - fric, 2),
        "roi_pct": round((prevented / (inv + fric) - 1) * 100, 1) if inv + fric else 0.0,
    }


def run(df: pd.DataFrame, seed: int, ae_epochs: int, measure_latency: bool) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.perf_counter()

    fe = FeatureEngineer()
    df_eng = fe.create_features(df)  # NOTE: statistics fit on the FULL dataset (incl. test)
    X = df_eng.drop(["Class", "Time"], axis=1)
    y = df_eng["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    # Duplicate leakage: exact copies of test rows that also appear in train.
    key_cols = [c for c in df.columns if c != "Class"]
    tr_keys = set(map(tuple, df.loc[X_tr.index, key_cols].to_numpy()))
    dup_in_test = int(sum(tuple(r) in tr_keys for r in df.loc[X_te.index, key_cols].to_numpy()))

    scaler = RobustScaler()
    Xs_tr, Xs_va, Xs_te = scaler.fit_transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    ae = FraudAutoencoder(Xs_tr.shape[1])
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xs_tr[y_tr.values == 0]), torch.FloatTensor(Xs_tr[y_tr.values == 0])),
        batch_size=256,
        shuffle=True,
    )
    opt = optim.Adam(ae.parameters(), lr=0.001, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
    losses = []
    ae.train()
    for _ in range(ae_epochs):
        tot = 0.0
        for bx, _ in loader:
            loss = nn.functional.mse_loss(ae(bx), bx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / len(loader))
        sched.step(losses[-1])

    iso = IsolationForest(
        n_estimators=200, max_samples="auto", contamination=0.002, random_state=seed, n_jobs=-1
    )
    iso.fit(Xs_tr)

    E_tr, E_va, E_te = (ensemble_features(a, ae, iso) for a in (Xs_tr, Xs_va, Xs_te))
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float((y_tr == 0).sum() / (y_tr == 1).sum()),
        random_state=seed,
        eval_metric="logloss",
        early_stopping_rounds=20,
        n_jobs=-1,
    )
    xgb.fit(E_tr, y_tr, eval_set=[(E_va, y_va)], verbose=False)

    proba = xgb.predict_proba(E_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    yt = y_te.to_numpy()
    amounts = df.loc[X_te.index, "Amount"].to_numpy()
    res: dict = {
        "seed": seed,
        "n_test": len(yt),
        "n_test_fraud": int(yt.sum()),
        "test_rows_with_exact_duplicate_in_train": dup_in_test,
        "roc_auc": float(roc_auc_score(yt, proba)),
        "pr_auc": float(average_precision_score(yt, proba)),
        "precision@0.5": float(precision_score(yt, pred)),
        "recall@0.5": float(recall_score(yt, pred)),
        "f1@0.5": float(f1_score(yt, pred)),
        "xgb_best_iteration": int(xgb.best_iteration),
        "ae_loss_first_last": [round(losses[0], 3), round(losses[-1], 3)] if losses else None,
        "ae_error_test_roc_auc": float(roc_auc_score(yt, E_te[:, -2])),
        "iso_score_test_roc_auc": float(roc_auc_score(yt, -E_te[:, -1])),
        "business_impact_flat_500": [business_impact(yt, proba, t) for t in (0.3, 0.5, 0.7, 0.9)],
        "business_impact_actual_amounts": [
            business_impact(yt, proba, t, amounts) for t in (0.3, 0.5, 0.7, 0.9)
        ],
        "train_seconds": round(time.perf_counter() - t0, 1),
    }

    if measure_latency:
        # The notebook's pipeline.predict() applied create_features() to rows of X_test, which are
        # ALREADY engineered. Reproduce that path and measure (a) latency and (b) train/serve skew.
        def legacy_predict(row: pd.DataFrame) -> float:
            feats = fe.create_features(row).drop(["Class", "Time"], axis=1, errors="ignore")
            xs = scaler.transform(feats)
            ae.eval()
            with torch.no_grad():
                err = ae.get_reconstruction_error(torch.FloatTensor(xs)).numpy()
            ens = np.column_stack([xs, err.reshape(-1, 1), iso.decision_function(xs).reshape(-1, 1)])
            return float(xgb.predict_proba(ens)[:, 1][0])

        lat, skew = [], []
        for i in range(200):
            row = X_te.iloc[i : i + 1]
            s = time.perf_counter()
            p_single = legacy_predict(row)
            lat.append((time.perf_counter() - s) * 1000)
            skew.append(abs(p_single - proba[i]))
        lat_a = np.array(lat[5:])  # drop warm-up
        res["legacy_pipeline_latency_ms"] = {
            "mean": round(float(lat_a.mean()), 2),
            "p50": round(float(np.percentile(lat_a, 50)), 2),
            "p95": round(float(np.percentile(lat_a, 95)), 2),
            "n": len(lat_a),
        }
        sk = np.array(skew)
        res["train_serve_skew"] = {
            "description": "|P(single-row legacy pipeline) - P(batch path)| on the same 200 test rows",
            "mean_abs_diff": float(sk.mean()),
            "max_abs_diff": float(sk.max()),
            "rows_with_diff_gt_0.01": int((sk > 0.01).sum()),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/creditcard.csv")
    ap.add_argument("--out", default="benchmarks/results/legacy_reproduction.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--ae-epochs", type=int, default=50)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    df = pd.read_csv(args.data)
    runs = []
    for i, seed in enumerate(args.seeds):
        r = run(df, seed, args.ae_epochs, measure_latency=(i == 0))
        print(
            json.dumps(
                {k: r[k] for k in ("seed", "roc_auc", "pr_auc", "precision@0.5", "recall@0.5", "f1@0.5")}
            )
        )
        runs.append(r)

    def spread(key: str) -> dict:
        v = np.array([r[key] for r in runs])
        return {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
        }

    out = {
        "description": "Faithful reproduction of notebooks/01_fraud_detection_complete.ipynb",
        "claimed": CLAIMED,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "xgboost": __import__("xgboost").__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "ae_epochs": args.ae_epochs,
        "runs": runs,
        "across_seeds": {
            k: spread(k) for k in ("roc_auc", "pr_auc", "precision@0.5", "recall@0.5", "f1@0.5")
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
