# Deployment

**Status:** fully working local and Docker deployment (verified, see `IMPLEMENTATION_PROGRESS.md`).
AWS: a documented, deliberately small target architecture with a task-definition template.
Nothing has been provisioned in AWS and no paid service is created by this repository.

## 1. Local (Python)

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/dev.txt && pip install --no-deps -e .
fraudshield download-data            # Kaggle CLI if configured, else public mirror; SHA-256 verified
fraudshield train --promote          # ~10 min on 8 CPU cores; or: --config configs/training_fast.yaml
cd dashboard && npm ci && npm run build && cd ..
uvicorn api.main:app --port 8000     # dashboard + API on http://localhost:8000, docs at /docs
```

**Second profile (IEEE-CIS v2, local only).** `FRAUDSHIELD_PROFILE=ieee` switches the request
contract, online velocity features, replay source and analyst text. Its paths default to
`artifacts/ieee/`, `var/fraudshield_ieee.db` and the replay/warm-up files written by
`fraudshield train-ieee`; each can be overridden (`FRAUDSHIELD_ARTIFACTS_DIR`, `FRAUDSHIELD_DB_PATH`,
`FRAUDSHIELD_REPLAY_FILE`, `FRAUDSHIELD_WARMUP_FILE`). The service refuses to start (readiness
fails with `RuntimeError`) if the profile does not match the artifact. The feature store is in
process, so this profile runs as a single instance. The IEEE-CIS data and models must not be hosted
publicly (competition rules); the Hugging Face Space uses the ULB profile.

## 1b. Public demo (Hugging Face Space)

Live at https://huggingface.co/spaces/notuearmand250/fraudshield. `deploy/hf-space/Dockerfile` trains the ULB model at build time and runs with
`FRAUDSHIELD_DEMO_MODE=true` (capped replays, no policy changes or resets) and
`FRAUDSHIELD_FRAME_ANCESTORS=https://huggingface.co`, so the Space page may embed the dashboard while
every other site is still denied. Publish or update it with
`scripts/deploy_hf_space.ps1 -SpaceId <user>/<space>`, using your own token. Docker Spaces need a
Hugging Face PRO plan (USD 9/month as of October 2026).

## 2. Docker

```bash
docker build --target runtime -t fraudshield-api:local .     # API + dashboard, no PyTorch
docker compose up -d api                                     # mounts ./artifacts read-only
docker compose --profile train run --rm trainer              # optional: train inside a container
docker compose --profile monitoring up -d                    # optional: Prometheus :9090, Grafana :3000
```

The runtime image runs as a non-root user; Compose adds a read-only root filesystem, a tmpfs
`/tmp`, `no-new-privileges`, and a named volume for the SQLite case store. The container health
check calls `GET /ready`, which fails until a checksum-verified model and the store are available.

## 3. Model promotion and rollback

```bash
fraudshield models list                                  # versions + test metrics
fraudshield models promote --version <v> --reason "..."  # runs the gate; refuses on failure
fraudshield models rollback --reason "..."               # previous production version
fraudshield evaluate                                     # re-score test with the production artifact
```

Gate checks (`configs/training.yaml → gate`): artifact integrity, deterministic smoke test, test
PR-AUC ≥ 0.60, test ROC-AUC ≥ 0.95, single-row p95 ≤ 50 ms, and no PR-AUC regression > 0.02 vs
the current production model when both were evaluated on the same data and split (otherwise the
gate refuses and a human must `--force` after review). Training never deploys automatically: the
API loads the registry's production pointer at start-up; pin `FRAUDSHIELD_MODEL_VERSION` to
override. Roll forward/back = change the pointer, restart (or redeploy) the service.

## 4. AWS target architecture (not provisioned)

```
            Route 53 ──► ALB (HTTPS, ACM cert) ──► ECS service on Fargate (1 task, 1 vCPU / 2 GB)
                                                     │  container: fraudshield-api (from ECR)
                                                     │  /ready health check, awslogs → CloudWatch Logs
                                                     │  secrets: FRAUDSHIELD_API_KEY from Secrets Manager
                              S3 (SSE-KMS, versioned, ◄┘  task role: s3:GetObject on models/* only
                              public access blocked)      init step syncs the production artifact
```

| Concern | Choice | Notes |
|---|---|---|
| Images | Amazon ECR, scan on push, immutable tags | Tag images with the git SHA; never `latest` in task definitions |
| Compute | ECS on Fargate, **one task** | The service keeps replay/SSE/rate-limit state in process. Scale out only after moving that state to ElastiCache (Redis) and the store to RDS Postgres |
| Artifacts | S3 bucket `models/<version>/…` + `registry.json` | Versioning on, KMS encryption, bucket policy limited to the task role. Artifacts are integrity-checked at load. Planned: an init container or entrypoint that runs `aws s3 sync` for the production version |
| Logs | CloudWatch Logs (`awslogs` driver) | JSON logs with request IDs; no feature values or analyst notes are logged |
| Metrics | CloudWatch Container Insights; optionally ADOT collector → Amazon Managed Prometheus | `/metrics` is Prometheus format |
| Secrets | Secrets Manager → task definition `secrets` | Never in the image or the repository |
| State | Ephemeral SQLite for a demo; RDS Postgres for anything real | SQLite on EFS is not recommended for concurrent writers |

A task-definition template lives in `deploy/aws/task-definition.json` (placeholders in `<…>`).
Typical commands, **to be run only with your explicit decision and AWS account**:

```bash
aws ecr create-repository --repository-name fraudshield-api --image-scanning-configuration scanOnPush=true
docker build --target runtime -t <acct>.dkr.ecr.<region>.amazonaws.com/fraudshield-api:<git-sha> .
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker push <acct>.dkr.ecr.<region>.amazonaws.com/fraudshield-api:<git-sha>
aws s3 sync artifacts/models/<version> s3://<bucket>/models/<version>
aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json
```

### Resource and cost considerations (estimates, verify with the AWS Pricing Calculator)

Measured locally in the Linux container: 135 MB memory at idle and 169 MB after a 12,000-request
load test; one uvicorn worker sustains ~240–290 scoring requests/s (p50 3.1 ms at one client; see
`ML_EVALUATION.md` §7). A single 1 vCPU / 2 GB Fargate task is therefore ample for demo traffic;
512 MB would likely suffice but leaves no headroom for the replay buffer and drift windows. At 2025 us-east-1 list prices the
rough monthly cost of running it 24/7 is: Fargate task ≈ $35, Application Load Balancer ≈ $20+,
ECR/S3/CloudWatch at demo volumes a few dollars. Stopping the service (desired count 0) removes the
compute cost; the ALB is billed while it exists.

## 5. Hardening checklist before any real deployment

- [ ] TLS everywhere; analyst SSO (OIDC) and role-based access in front of the dashboard
- [ ] `FRAUDSHIELD_API_KEY` (or mTLS / OAuth client credentials) for scoring clients
- [ ] Move state to Postgres + Redis; run ≥ 2 tasks across AZs
- [ ] Sign artifacts (e.g. Sigstore or KMS) in CI and verify signatures, not only checksums
- [ ] Data retention policy for stored features and notes; PII review of any new input fields
- [ ] Alerting on `/ready`, error rate, p95 latency, PSI, alert-rate shift
- [ ] Shadow-mode evaluation of new models on live traffic before promotion
