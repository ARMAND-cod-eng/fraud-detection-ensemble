"""Model artifact persistence, integrity checks and a file-based model registry.

Layout::

    artifacts/models/<version>/
        model.json            kind, feature builder, calibration, signal quantiles, linear weights
        booster.json          XGBoost model (tree kinds)
        autoencoder.npz       autoencoder weights for the NumPy forward pass
        iforest.joblib        Isolation Forest (pickle-based: loaded only after SHA-256 check)
        metadata.json         version, data hash, split, metrics, library versions, git commit
        reference_profile.json  drift-detection reference distributions
        evaluation.json       full evaluation report for this model
        MANIFEST.json         SHA-256 of every file above
    artifacts/registry.json   {"production": <version>, "history": [...]}
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from fraudshield.features.engineering import FeatureBuilder
from fraudshield.models.anomaly import IsolationForestDetector
from fraudshield.models.autoencoder import AutoencoderDetector
from fraudshield.models.calibration import PlattCalibrator
from fraudshield.models.fraud_model import FraudModel, LinearModel


class ArtifactIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def save_model(
    model: FraudModel,
    out_dir: str | Path,
    metadata: dict[str, Any],
    reference_profile: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an existing version
    spec: dict[str, Any] = {
        "kind": model.kind,
        "feature_builder": model.feature_builder.to_dict(),
        "calibrator": model.calibrator.to_dict(),
        "signal_quantiles": model.signal_quantiles,
        "model_feature_names": model.model_feature_names,
    }
    if model.booster is not None:
        model.booster.save_model(str(out / "booster.json"))
    if model.linear is not None:
        spec["linear"] = model.linear.to_dict()
    if model.autoencoder is not None:
        arrays: dict[str, Any] = model.autoencoder.to_npz_dict()
        np.savez(out / "autoencoder.npz", **arrays)
        spec["autoencoder_config"] = model.autoencoder.config_dict()
    if model.iforest is not None:
        joblib.dump(model.iforest, out / "iforest.joblib", compress=3)
    write_json(out / "model.json", spec)
    write_json(out / "metadata.json", metadata)
    if reference_profile is not None:
        write_json(out / "reference_profile.json", reference_profile)
    if evaluation is not None:
        write_json(out / "evaluation.json", evaluation)
    manifest = {p.name: _sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "MANIFEST.json", manifest)
    return out


def verify_artifact(model_dir: str | Path) -> None:
    d = Path(model_dir)
    manifest_path = d / "MANIFEST.json"
    if not manifest_path.exists():
        raise ArtifactIntegrityError(f"{d} has no MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        p = d / name
        if not p.exists() or _sha256(p) != digest:
            raise ArtifactIntegrityError(f"checksum mismatch for {name} in {d}")


def load_model(model_dir: str | Path, verify: bool = True) -> tuple[FraudModel, dict[str, Any]]:
    d = Path(model_dir)
    if verify:
        verify_artifact(d)  # must pass before anything pickle-based is deserialised
    spec = json.loads((d / "model.json").read_text(encoding="utf-8"))
    metadata = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    booster = None
    if (d / "booster.json").exists():
        booster = xgb.Booster()
        booster.load_model(str(d / "booster.json"))
    autoencoder = None
    if (d / "autoencoder.npz").exists():
        with np.load(d / "autoencoder.npz") as data:
            autoencoder = AutoencoderDetector.from_npz_dict(dict(data), spec.get("autoencoder_config"))
    iforest: IsolationForestDetector | None = None
    if (d / "iforest.joblib").exists():
        iforest = joblib.load(d / "iforest.joblib")
    model = FraudModel(
        kind=spec["kind"],
        feature_builder=FeatureBuilder.from_dict(spec["feature_builder"]),
        booster=booster,
        linear=LinearModel.from_dict(spec["linear"]) if "linear" in spec else None,
        iforest=iforest,
        autoencoder=autoencoder,
        calibrator=PlattCalibrator.from_dict(spec["calibrator"]),
        signal_quantiles=spec.get("signal_quantiles", {}),
    )
    model.prepare_for_serving(int(os.environ.get("FRAUDSHIELD_XGB_THREADS", "1")))
    return model, metadata


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- registry
class ModelRegistry:
    """Tiny file-based registry. Promotion and rollback are explicit, logged actions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.path = self.root / "registry.json"

    def _read(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"production": None, "history": []}

    def _write(self, reg: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        write_json(tmp, reg)
        tmp.replace(self.path)

    def versions(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(p.name for p in self.models_dir.iterdir() if (p / "MANIFEST.json").exists())

    def production(self) -> str | None:
        return self._read().get("production")

    def model_dir(self, version: str) -> Path:
        d = self.models_dir / version
        if not d.exists():
            raise FileNotFoundError(f"model version {version} not found in {self.models_dir}")
        return d

    def history(self) -> list[dict[str, Any]]:
        return list(self._read().get("history", []))

    def set_production(
        self, version: str, action: str, reason: str, gate: dict[str, Any] | None = None
    ) -> None:
        self.model_dir(version)
        reg = self._read()
        reg["history"].append(
            {
                "action": action,
                "version": version,
                "previous": reg.get("production"),
                "reason": reason,
                "gate": gate,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        reg["production"] = version
        self._write(reg)

    def rollback(self, reason: str = "manual rollback") -> str:
        reg = self._read()
        promotions = [h for h in reg["history"] if h["action"] in {"promote", "rollback"}]
        current = reg.get("production")
        for h in reversed(promotions):
            prev = h.get("previous")
            if h["version"] == current and prev:
                self.set_production(prev, "rollback", reason)
                return str(prev)
        raise RuntimeError("no previous production version to roll back to")
