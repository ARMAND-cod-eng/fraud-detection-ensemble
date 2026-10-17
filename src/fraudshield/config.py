"""Runtime settings, read from environment variables (prefix ``FRAUDSHIELD_``) or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUDSHIELD_", env_file=".env", extra="ignore")

    environment: str = "local"
    artifacts_dir: Path = Path("artifacts")
    model_version: str | None = Field(
        default=None, description="Pin a model version; default = registry production"
    )
    policy_path: Path = Path("configs/decision_policy.yaml")
    db_path: Path = Path("var/fraudshield.db")
    replay_file: Path = Path("artifacts/replay/creditcard_test_window.csv.gz")
    dashboard_dir: Path | None = Path("dashboard/dist")

    # Security: when set, mutating endpoints require header ``X-API-Key``.
    api_key: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    enable_docs: bool = True

    # Reliability
    inference_timeout_ms: int = Field(default=2000, ge=10)
    max_batch_size: int = Field(default=500, ge=1, le=10_000)
    rate_limit_per_second: float = Field(default=200.0, gt=0)
    rate_limit_burst: int = Field(default=400, ge=1)
    max_stored_predictions: int = Field(default=200_000, ge=1000)

    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
