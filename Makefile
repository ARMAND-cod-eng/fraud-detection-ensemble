# Convenience targets (POSIX shells). Windows users can run the commands in each recipe directly.
PY ?= python
.PHONY: install data train train-fast evaluate test lint format serve dashboard dashboard-dev docker up monitoring load-test reproduce-legacy

install:            ## dev environment (CPU PyTorch)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements/dev.txt
	$(PY) -m pip install --no-deps -e .

data:               ## download + SHA-256-verify the dataset
	fraudshield download-data

train:              ## full training + benchmark report + gated promotion (~10 min on 8 CPU cores)
	fraudshield train --promote

train-fast:         ## smaller grid on the real data (~3 min) for a quick demo model
	fraudshield train --config configs/training_fast.yaml --promote

evaluate:           ## reload the production artifact and re-score the test split
	fraudshield evaluate

reproduce-legacy:   ## faithful re-run of the original notebook (5 seeds)
	$(PY) benchmarks/reproduce_legacy.py --seeds 42 0 1 2 3

test:
	pytest -q

lint:
	ruff check . && ruff format --check . && mypy

format:
	ruff format . && ruff check --fix .

dashboard:          ## build the React dashboard into dashboard/dist (served by the API)
	cd dashboard && npm ci && npm run build

dashboard-dev:      ## hot-reload dashboard on :5173 (proxies to the API on :8000)
	cd dashboard && npm run dev

serve:
	uvicorn api.main:app --host 127.0.0.1 --port 8000

docker:
	docker build --target runtime -t fraudshield-api:local .

up:
	docker compose up -d api

monitoring:
	docker compose --profile monitoring up -d

load-test:          ## requires a running API
	$(PY) benchmarks/load_test.py --requests 2000 --concurrency 1 8 32
