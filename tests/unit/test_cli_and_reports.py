"""CLI, report generation, dataset download guards and optional tracking (no network, no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraudshield import cli
from fraudshield.data import download
from fraudshield.data.creditcard import sha256_file
from fraudshield.training import tracking
from fraudshield.training.report import markdown_report, write_report


def test_cli_models_list_show_evaluate_and_drift(trained, synthetic_csv, capsys) -> None:
    root = str(trained["root"])
    assert cli.main(["models", "list", "--artifacts", root]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["production"] == trained["version"]

    assert cli.main(["models", "show", "--artifacts", root]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == trained["version"]

    # reload the artifact, rebuild the split, re-score test: must reproduce the stored metrics
    assert cli.main(["evaluate", "--artifacts", root, "--data", str(synthetic_csv)]) == 0
    assert json.loads(capsys.readouterr().out)["reproduced"] is True

    assert cli.main(["drift", "--csv", str(synthetic_csv), "--artifacts", root]) == 0
    assert "max_feature_psi" in json.loads(capsys.readouterr().out)


def test_cli_train_gate_and_rollback(tmp_path, synthetic_csv, capsys, monkeypatch) -> None:
    import yaml

    monkeypatch.setenv("FRAUDSHIELD_DISABLE_MLFLOW", "1")
    arts = str(tmp_path / "a")
    args = [
        "train",
        "--quick",
        "--data",
        str(synthetic_csv),
        "--artifacts",
        arts,
        "--results",
        str(tmp_path / "r"),
        "--no-figures",
    ]
    assert cli.main(args) == 0  # trains without promoting
    out = capsys.readouterr().out  # progress lines, then the JSON summary
    version = json.loads(out[out.index('{\n  "version"') :])["version"]
    assert (tmp_path / "r" / "report.md").exists()

    # A gate tightened after training applies at promotion time and refuses the model.
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "training.yaml").read_text())
    cfg["gate"]["min_test_pr_auc"] = 1.01
    strict = tmp_path / "strict.yaml"
    strict.write_text(yaml.safe_dump(cfg))
    promote = ["models", "promote", "--version", version, "--artifacts", arts]
    assert cli.main([*promote, "--config", str(strict)]) == 1
    capsys.readouterr()
    assert cli.main(promote) == 0  # default gate passes

    # Only one promotion so far: nothing to roll back to, reported as an error, not a crash.
    assert cli.main(["models", "rollback", "--artifacts", arts]) == 1


def test_report_markdown_and_figures(trained, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    ev = trained["evaluation"]
    md = markdown_report(ev)
    assert "## Test results (scored once)" in md and ev["version"] in md
    write_report(ev, tmp_path, make_figures=True)
    assert json.loads((tmp_path / "metrics.json").read_text())["version"] == ev["version"]
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {
        "pr_curves.png",
        "roc_curves.png",
        "calibration.png",
        "cost_curve.png",
        "global_importance.png",
    } <= figures


def test_download_uses_verified_local_file_and_refuses_bad_ones(tmp_path, monkeypatch) -> None:
    good = tmp_path / "creditcard.csv"
    good.write_text("Time,V1\n0,1\n")
    monkeypatch.setattr(download, "EXPECTED_SHA256", sha256_file(good))
    assert download.ensure_creditcard(good) == good  # verified local copy: no network

    bad = tmp_path / "other" / "creditcard.csv"
    bad.parent.mkdir()
    bad.write_text("tampered\n")
    monkeypatch.setattr(download.shutil, "which", lambda _: None)  # no Kaggle CLI
    with pytest.raises(RuntimeError, match="Dataset not found"):
        download.ensure_creditcard(bad, allow_public_mirror=False)


def test_tracking_is_optional(trained, monkeypatch) -> None:
    cfg = {"tracking": {"mlflow": True, "uri": "mlruns"}, "experiment_name": "x", "seed": 0}
    monkeypatch.setenv("FRAUDSHIELD_DISABLE_MLFLOW", "1")
    assert tracking.log_run(cfg, trained["evaluation"], Path(trained["model_dir"]), None) is False
    monkeypatch.delenv("FRAUDSHIELD_DISABLE_MLFLOW")
    assert (
        tracking.log_run({**cfg, "tracking": {"mlflow": False}}, trained["evaluation"], Path("."), None)
        is False
    )
    flat = tracking._flat_metrics(trained["evaluation"])
    assert any(k.startswith("test.xgboost.") for k in flat) and all(
        isinstance(v, float) for v in flat.values()
    )
    assert tracking._resolve_uri("mlruns").startswith("file:")
    assert tracking._resolve_uri("http://tracking:5000") == "http://tracking:5000"
