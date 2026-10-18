"""Online/offline parity check of the velocity features on the real IEEE-CIS data.

1. Offline: compute every velocity feature over the full time-ordered history (training path).
2. Online: warm an ``OnlineFeatureStore`` with the events in the max window (7 days) before the
   test window, then stream the test window event by event (``score_and_update``), as the API does.
3. Compare. Expected result: zero mismatches (tolerance 1e-6 absolute, 1e-9 relative).

    python scripts/check_feature_parity.py --root data/raw/ieee-cis --out benchmarks/results/ieee-cis-v2/parity.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fraudshield.data.ieee_cis import load_contract
from fraudshield.features.velocity import VELOCITY_SPECS, OnlineFeatureStore, compute_offline
from fraudshield.profiles.ieee import CONTRACT_FIELDS, TIME


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw/ieee-cis")
    ap.add_argument(
        "--test-frac", type=float, default=0.2, help="final share of the timeline to stream online"
    )
    ap.add_argument("--out", default="benchmarks/results/ieee-cis-v2/parity.json")
    args = ap.parse_args()

    df = load_contract(args.root)
    t0 = time.perf_counter()
    offline = compute_offline(df)
    offline_s = time.perf_counter() - t0

    start = int(len(df) * (1 - args.test_frac))
    window = max(s.window_s for s in VELOCITY_SPECS)
    t_start = df[TIME].iloc[start]
    warm = df.iloc[:start][df[TIME].iloc[:start] >= t_start - window]
    cols = [TIME, *CONTRACT_FIELDS]
    store = OnlineFeatureStore()
    t1 = time.perf_counter()
    store.warm_up(warm[cols].to_dict("records"))
    online = [store.score_and_update(r) for r in df.iloc[start:][cols].to_dict("records")]
    online_s = time.perf_counter() - t1

    names = [s.name for s in VELOCITY_SPECS]
    on = np.array([[r[n] for n in names] for r in online], dtype=np.float64)
    off = offline.iloc[start:][names].to_numpy(dtype=np.float64)
    both_nan = np.isnan(on) & np.isnan(off)
    close = np.isclose(on, off, rtol=1e-9, atol=1e-6) | both_nan
    per_feature = {n: int((~close[:, j]).sum()) for j, n in enumerate(names)}
    report = {
        "rows_streamed_online": len(online),
        "warm_up_events": len(warm),
        "features": len(names),
        "values_compared": int(close.size),
        "mismatches": int((~close).sum()),
        "mismatches_per_feature": per_feature,
        "offline_seconds_full_history": round(offline_s, 1),
        "online_seconds_warm_up_plus_stream": round(online_s, 1),
        "online_microseconds_per_event": round(online_s / max(1, len(online) + len(warm)) * 1e6, 1),
        "tolerance": {"rtol": 1e-9, "atol": 1e-6},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
