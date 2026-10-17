"""End-to-end HTTP load test for POST /api/v1/transactions/score.

Measures client-observed latency (loopback network + validation + model + policy + SQLite write
+ telemetry), not just model time. Payloads are real held-out transactions from the replay file.

Each concurrent client is a separate *process* holding one keep-alive ``http.client`` connection.
(An asyncio client in a single process added ~5 ms of client-side overhead per request on
Windows and distorted the numbers; separate processes keep the client out of the measurement.)

    python benchmarks/load_test.py --url http://127.0.0.1:8000 --requests 3000 --concurrency 1 4 8 16

Results depend on hardware and server configuration (one uvicorn worker by default); the JSON
output records both. The client runs on the same machine and competes for the same CPU cores.
"""

from __future__ import annotations

import argparse
import http.client
import json
import multiprocessing as mp
import os
import platform
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # many client processes on one machine
import numpy as np
import pandas as pd

V = [f"V{i}" for i in range(1, 29)]
PATH = "/api/v1/transactions/score"


def payloads(path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(path).sample(n=n, replace=True, random_state=seed)
    return [
        {
            "amount": float(r.Amount),
            "time_seconds": float(r.Time),
            "features": {v: float(getattr(r, v)) for v in V},
        }
        for r in df.itertuples(index=False)
    ]


def _client(args: tuple[str, list[dict], float]) -> tuple[list[float], dict[str, int], float, float]:
    url, bodies, start_at = args
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    headers = {"Content-Type": "application/json"}
    for b in bodies[:10]:  # warm-up, not measured
        conn.request("POST", PATH, json.dumps(b), headers)
        conn.getresponse().read()
    while time.time() < start_at:  # start all clients together
        time.sleep(0.001)
    lat: list[float] = []
    errors: dict[str, int] = {}
    t0 = time.perf_counter()
    for b in bodies:
        body = json.dumps({**b, "transaction_id": f"lt-{uuid.uuid4().hex[:16]}"})
        t = time.perf_counter()
        try:
            conn.request("POST", PATH, body, headers)
            r = conn.getresponse()
            r.read()
            if r.status == 200:
                lat.append((time.perf_counter() - t) * 1000)
            else:
                errors[str(r.status)] = errors.get(str(r.status), 0) + 1
        except OSError as exc:
            errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    return lat, errors, t0, time.perf_counter()


def run_level(url: str, bodies: list[dict], concurrency: int) -> dict:
    chunks = [bodies[i::concurrency] for i in range(concurrency)]
    start_at = time.time() + 3.0
    with mp.Pool(concurrency) as pool:
        results = pool.map(_client, [(url, c, start_at) for c in chunks])
    lat = np.asarray([x for r in results for x in r[0]])
    errors: dict[str, int] = {}
    for r in results:
        for k, v in r[1].items():
            errors[k] = errors.get(k, 0) + v
    wall = max(r[3] for r in results) - min(r[2] for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(bodies),
        "ok": len(lat),
        "errors": errors,
        "throughput_rps": len(lat) / wall if wall else 0.0,
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "mean": float(lat.mean()),
            "max": float(lat.max()),
        }
        if len(lat)
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--replay-file", default="artifacts/replay/creditcard_test_window.csv.gz")
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="local")
    ap.add_argument("--out", default="benchmarks/results/load_test.json")
    args = ap.parse_args()

    u = urlparse(args.url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5)
    c.request("GET", "/ready")
    ready = json.loads(c.getresponse().read())
    bodies = payloads(Path(args.replay_file), args.requests, args.seed)
    levels = []
    for conc in args.concurrency:
        res = run_level(args.url, bodies, conc)
        print(json.dumps({k: res[k] for k in ("concurrency", "throughput_rps", "latency_ms", "errors")}))
        levels.append(res)
    out = {
        "label": args.label,
        "endpoint": f"POST {PATH}",
        "model_version": ready.get("model_version"),
        "server": "uvicorn, 1 worker, same machine as the client (loopback)",
        "client": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "method": "one process + one keep-alive connection per concurrent client",
        },
        "payloads": f"{args.requests} real held-out transactions (seed {args.seed}), unique IDs",
        "levels": levels,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
