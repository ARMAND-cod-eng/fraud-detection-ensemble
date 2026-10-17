"""Capture dashboard screenshots with a headless Chromium-based browser (Edge or Chrome).

The API must be running with data in it (e.g. after a replay). Uses a throw-away browser profile.

    python scripts/capture_screenshots.py --url http://localhost:8000 --out docs/images \
        --pages overview live performance policy monitoring "cases/<transaction_id>"

Set BROWSER_PATH if the browser is not found automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CANDIDATES = [
    os.environ.get("BROWSER_PATH", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "msedge",
]


def find_browser() -> str:
    for c in CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    edge_core = Path(r"C:\Program Files (x86)\Microsoft\EdgeCore")
    if edge_core.exists():
        for exe in sorted(edge_core.glob("*/msedge.exe"), reverse=True):
            return str(exe)
    raise SystemExit("no Chromium-based browser found; set BROWSER_PATH")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--pages", nargs="+", default=["overview", "live", "performance", "policy", "monitoring"])
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--wait-ms", type=int, default=6000)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    args = ap.parse_args()

    browser = find_browser()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        for page in args.pages:
            name = page.split("/")[0] + ("_dark" if args.theme == "dark" else "")
            target = out / f"{name}.png"
            cmd = [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                f"--user-data-dir={profile}",
                f"--window-size={args.width},{args.height}",
                f"--screenshot={target}",
                # pages with an open SSE stream never go idle, so they use a hard timeout instead
                f"--timeout={args.wait_ms}"
                if page.startswith("live")
                else f"--virtual-time-budget={args.wait_ms}",
                *(
                    ["--force-dark-mode", "--blink-settings=preferredColorScheme=0"]
                    if args.theme == "dark"
                    else []
                ),
                f"{args.url}/#/{page}",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=args.wait_ms / 1000 + 60)
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
