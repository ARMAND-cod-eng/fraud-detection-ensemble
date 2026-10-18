"""End-to-end parity through the running API: replayed transactions vs. offline training features.

``check_feature_parity.py`` compares the two feature engines directly. This script checks the whole
serving path instead: a replay is streamed through HTTP -> FraudService -> online feature store ->
SQLite, then the velocity values *stored with each prediction* are compared with the offline
features computed over warm-up + test window (the exact training-time definition).

Prerequisites: an API running with ``FRAUDSHIELD_PROFILE=ieee`` on a fresh database (reset it with
``POST /api/v1/admin/reset`` and restart, so the stream starts at row 0 of the test window).

    python scripts/check_served_parity.py --api http://127.0.0.1:8010 --count 5000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.features.velocity import VELOCITY_SPECS, compute_offline


def _call(base: str, path: str, body: dict[str, Any] | None = None) -> Any:
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8010")
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--rate", type=float, default=400)
    ap.add_argument("--artifacts", default="artifacts/ieee")
    ap.add_argument("--out", default="benchmarks/results/ieee-cis-v2/served_parity.json")
    args = ap.parse_args()
    api = args.api.rstrip("/") + "/api/v1"

    start = _call(
        api, "/replay/start", {"scenario": "historical", "rate_tps": args.rate, "count": args.count}
    )
    first_row = int(start["notes"][1].split("rows ")[1].split("-")[0].replace(",", ""))
    while (st := _call(api, "/replay/status"))["state"] == "running":
        time.sleep(1)
    if st["state"] != "completed" or st["failed"]:
        raise SystemExit(f"replay did not complete cleanly: {st['state']}, failed={st['failed']}")

    replay_dir = Path(args.artifacts) / "replay"
    warm = pd.read_csv(replay_dir / "ieee_warmup.csv.gz")
    test = pd.read_csv(replay_dir / "ieee_test_window.csv.gz")
    offline = compute_offline(pd.concat([warm, test], ignore_index=True)).iloc[len(warm) :]
    names = [s.name for s in VELOCITY_SPECS]
    expected = offline.iloc[first_row : first_row + st["scored"]][names].to_numpy(dtype=np.float64)

    served = np.empty_like(expected)
    for i in range(st["scored"]):
        f = _call(api, f"/transactions/rp-{st['run_id']}-{i:06d}")["features"]
        served[i] = [np.nan if f[k] is None else f[k] for k in names]
    ok = np.isclose(served, expected, rtol=1e-9, atol=1e-6) | (np.isnan(served) & np.isnan(expected))
    report = {
        "path": "HTTP replay -> FraudService -> OnlineFeatureStore -> SQLite (stored features)",
        "test_window_rows": f"{first_row}-{first_row + st['scored'] - 1}",
        "transactions": int(st["scored"]),
        "values_compared": int(ok.size),
        "mismatches": int((~ok).sum()),
        "mismatches_per_feature": {k: int((~ok[:, j]).sum()) for j, k in enumerate(names)},
        "replay_effective_tps": round(st["effective_tps"], 1),
        "tolerance": {"rtol": 1e-9, "atol": 1e-6},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
