# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------- dashboard build
FROM node:22-alpine AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# ---------------------------------------------------------------- API runtime (no PyTorch)
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
# xgboost-cpu provides the same `xgboost` module; skip dependency resolution to keep it.
RUN pip install --no-deps . && rm -rf src api
COPY configs ./configs
COPY --from=dashboard /dashboard/dist ./dashboard/dist
RUN mkdir -p /app/var /app/artifacts && chown -R app:app /app/var
ENV FRAUDSHIELD_ARTIFACTS_DIR=/app/artifacts \
    FRAUDSHIELD_DB_PATH=/app/var/fraudshield.db \
    FRAUDSHIELD_DASHBOARD_DIR=/app/dashboard/dist \
    FRAUDSHIELD_REPLAY_FILE=/app/artifacts/replay/creditcard_test_window.csv.gz \
    FRAUDSHIELD_ENVIRONMENT=docker
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"
# One worker: replay state, SSE fan-out and rate limits are in-process (see ENGINEERING_TRADEOFFS.md).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

# ---------------------------------------------------------------- training image (adds PyTorch CPU)
FROM runtime AS trainer
USER root
COPY requirements/train.txt requirements/train.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install matplotlib==3.11.2
COPY benchmarks ./benchmarks
RUN mkdir -p /app/data && chown -R app:app /app/data /app/artifacts /app/benchmarks
USER app
ENV FRAUDSHIELD_DISABLE_MLFLOW=1
CMD ["sh", "-c", "fraudshield download-data && fraudshield train --promote"]
