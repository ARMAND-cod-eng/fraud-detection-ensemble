"""Autoencoder anomaly detector.

Fixes relative to the legacy notebook:
* Inputs are standardised with statistics from *training normals only* and clipped to +/-10
  standard deviations, instead of RobustScaler output fed into a Sigmoid output layer. The
  legacy model's targets were unbounded (e.g. Amount squared) while its outputs were bounded
  to (0, 1), so it could not reconstruct anything: the loss stayed at ~1418 for 50 epochs.
* Linear output layer, no BatchNorm/Dropout (simpler, deterministic, exportable).
* Early stopping on a held-out slice of training normals.

PyTorch is used only for training. Inference runs in NumPy from exported weights, so the
serving image does not need PyTorch. A unit test checks both forward passes agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CLIP = 10.0


@dataclass
class AutoencoderConfig:
    hidden: tuple[int, ...] = (32, 16)
    latent: int = 8
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 4
    holdout_frac: float = 0.1
    seed: int = 42


@dataclass
class AutoencoderDetector:
    config: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    weights_: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    history_: dict[str, list[float]] = field(default_factory=dict)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return np.clip((x - self.mean_) / self.std_, -CLIP, CLIP)

    def fit(self, x_normal: np.ndarray) -> AutoencoderDetector:
        import torch
        from torch import nn

        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.mean_ = x_normal.mean(axis=0)
        self.std_ = x_normal.std(axis=0) + 1e-8
        z = self._standardise(x_normal).astype(np.float32)
        idx = rng.permutation(len(z))
        n_hold = max(1, int(len(z) * cfg.holdout_frac))
        z_val, z_tr = torch.from_numpy(z[idx[:n_hold]]), torch.from_numpy(z[idx[n_hold:]])

        dims = [z.shape[1], *cfg.hidden, cfg.latent, *reversed(cfg.hidden), z.shape[1]]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(cfg.seed)

        best, best_state, bad = float("inf"), None, 0
        self.history_ = {"train_loss": [], "holdout_loss": []}
        for _ in range(cfg.epochs):
            model.train()
            perm = torch.randperm(len(z_tr), generator=gen)
            total = 0.0
            for s in range(0, len(z_tr), cfg.batch_size):
                batch = z_tr[perm[s : s + cfg.batch_size]]
                loss = loss_fn(model(batch), batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            model.eval()
            with torch.no_grad():
                hold = float(loss_fn(model(z_val), z_val))
            self.history_["train_loss"].append(total / len(z_tr))
            self.history_["holdout_loss"].append(hold)
            if hold < best - 1e-5:
                best, bad = hold, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.weights_ = [
            (
                m.weight.detach().numpy().astype(np.float64).T.copy(),
                m.bias.detach().numpy().astype(np.float64).copy(),
            )
            for m in model
            if isinstance(m, nn.Linear)
        ]
        return self

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        h = self._standardise(np.asarray(x, dtype=np.float64))
        for i, (w, b) in enumerate(self.weights_):
            h = h @ w + b
            if i < len(self.weights_) - 1:
                h = np.maximum(h, 0.0)
        return h

    def score(self, x: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error in standardised space (higher = more anomalous)."""
        z = self._standardise(np.asarray(x, dtype=np.float64))
        return ((self.reconstruct(x) - z) ** 2).mean(axis=1)

    # --- persistence ---------------------------------------------------------------------
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        assert self.mean_ is not None and self.std_ is not None
        out: dict[str, np.ndarray] = {"mean": self.mean_, "std": self.std_}
        for i, (w, b) in enumerate(self.weights_):
            out[f"w{i}"], out[f"b{i}"] = w, b
        return out

    @classmethod
    def from_npz_dict(cls, data: Any, config: dict[str, Any] | None = None) -> AutoencoderDetector:
        kwargs: dict[str, Any] = dict(config or {})
        if "hidden" in kwargs:
            kwargs["hidden"] = tuple(kwargs["hidden"])
        cfg = AutoencoderConfig(**kwargs)
        det = cls(config=cfg, mean_=np.asarray(data["mean"]), std_=np.asarray(data["std"]))
        n_layers = sum(1 for k in data if str(k).startswith("w"))
        det.weights_ = [(np.asarray(data[f"w{i}"]), np.asarray(data[f"b{i}"])) for i in range(n_layers)]
        return det

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
