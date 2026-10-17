"""Optional MLflow experiment tracking (no-op when MLflow is not installed or disabled)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


def _flat_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    sel = evaluation["selected_model"]
    for split in ("validation", "test"):
        for model, metrics in evaluation[split].items():
            for k, v in metrics.items():
                if isinstance(v, int | float):
                    key = f"{split}.{model}.{k}".replace("@", "_at_").replace("=", "_").replace("%", "pct")
                    key = "".join(ch if ch.isalnum() or ch in "._-/ " else "_" for ch in key)
                    out[key] = float(v)
    op = evaluation["selected_operating_points"]["review_threshold_test"]
    out.update({f"selected.review.{k}": float(v) for k, v in op.items() if isinstance(v, int | float)})
    out[f"selected.latency_p95_ms.{sel}"] = evaluation["latency_in_process"][sel]["single_row_p95_ms"]
    return out


def _resolve_uri(uri: str) -> str:
    """A bare path becomes a file:// URI (works with both mlflow and mlflow-skinny)."""
    return uri if "://" in uri or uri.startswith("file:") else Path(uri).resolve().as_uri()


def log_run(
    cfg: dict[str, Any], evaluation: dict[str, Any], model_dir: Path, results_dir: Path | str | None
) -> bool:
    """Best-effort experiment tracking: a tracking failure never fails a training run."""
    if not cfg.get("tracking", {}).get("mlflow", False) or os.environ.get("FRAUDSHIELD_DISABLE_MLFLOW"):
        return False
    try:
        import mlflow
    except ImportError:
        return False
    try:
        _log(mlflow, cfg, evaluation, model_dir, results_dir)
        return True
    except Exception as exc:  # tracking is auxiliary; the artifact and report are already saved
        warnings.warn(
            f"MLflow tracking failed ({type(exc).__name__}: {exc}); training results are unaffected",
            stacklevel=2,
        )
        return False


def _log(
    mlflow: Any,
    cfg: dict[str, Any],
    evaluation: dict[str, Any],
    model_dir: Path,
    results_dir: Path | str | None,
) -> None:
    uri = _resolve_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["tracking"].get("uri", "mlruns")))
    if uri.startswith("file:"):
        # Local, zero-infrastructure store. MLflow 3 keeps the file backend in maintenance mode and
        # requires an explicit opt-in; point MLFLOW_TRACKING_URI at a server/database to avoid it.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=evaluation["version"]):
        mlflow.set_tags(
            {
                "selected_model": evaluation["selected_model"],
                "data_sha256": evaluation["dataset"]["sha256"],
                "git_commit": str(evaluation["environment"].get("git_commit")),
            }
        )
        cand = evaluation["candidates"]["xgboost"]["info"]["params"]
        mlflow.log_params({f"xgb.{k}": v for k, v in cand.items()})
        mlflow.log_params(
            {
                "split": evaluation["dataset"]["split"]["strategy"],
                "seed": cfg["seed"],
                "review_threshold": evaluation["selected_operating_points"]["review_threshold"],
            }
        )
        mlflow.log_metrics(_flat_metrics(evaluation))
        for f in ("metadata.json", "evaluation.json", "MANIFEST.json"):
            mlflow.log_artifact(str(model_dir / f), artifact_path="model")
        if results_dir is not None and Path(results_dir).exists():
            mlflow.log_artifacts(str(results_dir), artifact_path="report")
