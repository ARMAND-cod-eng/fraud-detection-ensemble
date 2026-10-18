"""Runtime settings, read from environment variables (prefix ``FRAUDSHIELD_``) or a .env file."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUDSHIELD_", env_file=".env", extra="ignore")

    environment: str = "local"
    # Data profile served: "creditcard" (ULB) or "ieee" (IEEE-CIS v2, authorisation-time fields +
    # online velocity features). Paths below default per profile unless set explicitly.
    profile: Literal["creditcard", "ieee"] = "creditcard"
    artifacts_dir: Path = Path("artifacts")
    model_version: str | None = Field(
        default=None, description="Pin a model version; default = registry production"
    )
    policy_path: Path = Path("configs/decision_policy.yaml")
    db_path: Path = Path("var/fraudshield.db")
    replay_file: Path = Path("artifacts/replay/creditcard_test_window.csv.gz")
    warmup_file: Path | None = None  # history that warms the online feature store (ieee profile)
    dashboard_dir: Path | None = Path("dashboard/dist")

    # Security: when set, mutating endpoints require header ``X-API-Key``.
    api_key: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Origins allowed to embed the dashboard in a frame (space-separated https origins), e.g.
    # "https://huggingface.co" for a Space page. Empty (default): framing is denied everywhere.
    frame_ancestors: str = ""
    enable_docs: bool = True

    # Reliability
    inference_timeout_ms: int = Field(default=2000, ge=10)
    max_batch_size: int = Field(default=500, ge=1, le=10_000)
    rate_limit_per_second: float = Field(default=200.0, gt=0)
    rate_limit_burst: int = Field(default=400, ge=1)
    max_stored_predictions: int = Field(default=200_000, ge=1000)

    log_level: str = "INFO"
    log_json: bool = True

    # Public demo (e.g. a Hugging Face Space): caps replay load and disables state-changing admin
    # actions (policy changes, reset). What-if simulation and analyst reviews stay available.
    demo_mode: bool = False
    demo_max_replay_tps: float = Field(default=100.0, gt=0)
    demo_max_replay_count: int = Field(default=5000, ge=1)
    demo_note: str = ""  # shown in the dashboard banner, e.g. how the demo model was trained

    @field_validator("frame_ancestors")
    @classmethod
    def _check_frame_ancestors(cls, v: str) -> str:
        # Values go verbatim into a response header: allow only plain https origins.
        for origin in v.split():
            if not re.fullmatch(r"https://[A-Za-z0-9.-]+(:\d{1,5})?", origin):
                raise ValueError(f"frame_ancestors entries must be https origins, got {origin!r}")
        return " ".join(v.split())

    @model_validator(mode="after")
    def _profile_defaults(self) -> Settings:
        if self.profile == "ieee":
            defaults = {
                "artifacts_dir": Path("artifacts/ieee"),
                "db_path": Path("var/fraudshield_ieee.db"),
                "replay_file": Path("artifacts/ieee/replay/ieee_test_window.csv.gz"),
                "warmup_file": Path("artifacts/ieee/replay/ieee_warmup.csv.gz"),
            }
            for name, value in defaults.items():
                if name not in self.model_fields_set:
                    object.__setattr__(self, name, value)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
