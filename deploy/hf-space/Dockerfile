# syntax=docker/dockerfile:1.7
# Hugging Face Space image for the public FraudShield demo (Docker SDK, free CPU tier).
# Nothing pre-trained is shipped: the build downloads the SHA-256-verified public dataset and runs
# the documented training pipeline. Built by scripts/deploy_hf_space.ps1 from a `git archive`.

# ---------------------------------------------------------------- dashboard
FROM node:22-alpine AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# ---------------------------------------------------------------- train at build time
FROM python:3.12-slim AS trainer
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 FRAUDSHIELD_DISABLE_MLFLOW=1
WORKDIR /build
COPY requirements/ requirements/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements/train.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
COPY configs ./configs
RUN pip install --no-deps .
RUN fraudshield download-data \
 && fraudshield train --config configs/training_fast.yaml --promote --no-figures \
 && rm -rf data

# ---------------------------------------------------------------- serve
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
RUN pip install --no-deps . && rm -rf src api
COPY configs ./configs
COPY --from=dashboard /dashboard/dist ./dashboard/dist
COPY --from=trainer /build/artifacts ./artifacts
ENV FRAUDSHIELD_ENVIRONMENT=hf-space \
    FRAUDSHIELD_DEMO_MODE=true \
    FRAUDSHIELD_DEMO_NOTE="Demo model trained when this Space was built (configs/training_fast.yaml: same protocol, smaller search); the GitHub README reports the full benchmark." \
    FRAUDSHIELD_ARTIFACTS_DIR=/app/artifacts \
    FRAUDSHIELD_REPLAY_FILE=/app/artifacts/replay/creditcard_test_window.csv.gz \
    FRAUDSHIELD_DASHBOARD_DIR=/app/dashboard/dist \
    FRAUDSHIELD_DB_PATH=/tmp/fraudshield.db \
    FRAUDSHIELD_MAX_STORED_PREDICTIONS=20000 \
    FRAUDSHIELD_LOG_LEVEL=WARNING
USER user
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
