from __future__ import annotations

import json

import numpy as np
import pytest

from fraudshield.data.synthetic import generate_creditcard_like
from fraudshield.features.engineering import FeatureBuilder
from fraudshield.models.artifact import ArtifactIntegrityError, ModelRegistry, load_model, verify_artifact
from fraudshield.models.autoencoder import AutoencoderConfig, AutoencoderDetector
from fraudshield.models.calibration import PlattCalibrator, logit


def test_autoencoder_numpy_forward_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(0).normal(size=(400, 12))
    ae = AutoencoderDetector(AutoencoderConfig(hidden=(8,), latent=4, epochs=2, batch_size=64)).fit(x)
    layers = []
    for i, (w, b) in enumerate(ae.weights_):
        lin = torch.nn.Linear(w.shape[0], w.shape[1]).double()
        lin.weight.data = torch.from_numpy(w.T.copy())
        lin.bias.data = torch.from_numpy(b.copy())
        layers.append(lin)
        if i < len(ae.weights_) - 1:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    z = ae._standardise(x)
    with torch.no_grad():
        expected = net(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(ae.reconstruct(x), expected, atol=1e-9)
    restored = AutoencoderDetector.from_npz_dict(ae.to_npz_dict(), ae.config_dict())
    np.testing.assert_allclose(restored.score(x), ae.score(x))


def test_autoencoder_scores_anomalies_higher() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    normal = rng.normal(size=(3000, 10))
    ae = AutoencoderDetector(AutoencoderConfig(hidden=(8,), latent=3, epochs=15, batch_size=128)).fit(normal)
    anomalies = rng.normal(size=(200, 10)) + 4.0
    assert ae.score(anomalies).mean() > 3 * ae.score(normal[:200]).mean()


def test_platt_calibration_fixes_overconfident_scores() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.0, 0.2, 20_000)
    y = (rng.random(20_000) < true_p).astype(int)
    overconfident = 1 / (1 + np.exp(-(3 * logit(true_p) + 2)))
    cal = PlattCalibrator().fit(overconfident, y)
    brier_raw = np.mean((overconfident - y) ** 2)
    brier_cal = np.mean((cal.transform(overconfident) - y) ** 2)
    assert brier_cal < brier_raw
    order = np.argsort(overconfident)
    assert np.all(np.diff(cal.transform(overconfident)[order]) >= -1e-12)  # monotone: ranking preserved
    assert (
        PlattCalibrator.from_dict(cal.to_dict()).transform(np.array([0.3]))[0]
        == cal.transform(np.array([0.3]))[0]
    )


def test_contributions_sum_to_calibrated_logit(trained) -> None:
    model, _ = load_model(trained["model_dir"])
    raw = generate_creditcard_like(n=50, seed=3)
    x = model.model_inputs(model.base_features(raw))
    contrib, base = model.contributions(x)
    p = model.predict_from_inputs(x)
    np.testing.assert_allclose(contrib.sum(axis=1) + base, logit(p), rtol=1e-4, atol=1e-4)


def test_artifact_roundtrip_and_tamper_detection(trained, tmp_path) -> None:
    import shutil

    model, _ = load_model(trained["model_dir"])
    raw = generate_creditcard_like(n=100, seed=4)
    p = model.predict_proba(raw)
    copy = tmp_path / "copy"
    shutil.copytree(trained["model_dir"], copy)
    model2, _ = load_model(copy)
    np.testing.assert_array_equal(model2.predict_proba(raw), p)

    spec = json.loads((copy / "model.json").read_text())
    spec["calibrator"]["a"] = 99.0
    (copy / "model.json").write_text(json.dumps(spec))
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(copy)
    with pytest.raises(ArtifactIntegrityError):
        load_model(copy)


def test_registry_promote_and_rollback(trained, tmp_path) -> None:
    import shutil

    from fraudshield.training.promotion import PromotionRejected, promote_version

    root = tmp_path / "reg"
    v = trained["version"]
    shutil.copytree(trained["model_dir"], root / "models" / v)
    v2 = v + "-b"
    shutil.copytree(trained["model_dir"], root / "models" / v2)
    reg = ModelRegistry(root)
    gate = {"max_pr_auc_regression": 0.02}
    promote_version(reg, v, gate)
    promote_version(reg, v2, gate)
    assert reg.production() == v2
    assert reg.rollback("test") == v
    assert reg.production() == v
    assert [h["action"] for h in reg.history()] == ["promote", "promote", "rollback"]

    # a model whose recorded gate checks fail must not be promoted
    meta_path = root / "models" / v2 / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["gate_checks"]["test_pr_auc"] = [0.1, ">=", 0.6, False]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PromotionRejected):
        promote_version(reg, v2, gate)  # also fails integrity because metadata changed


def test_feature_builder_rejects_incompatible_artifact() -> None:
    d = FeatureBuilder().to_dict()
    d["feature_names"] = d["feature_names"][:-1]
    with pytest.raises(ValueError):
        FeatureBuilder.from_dict(d)
