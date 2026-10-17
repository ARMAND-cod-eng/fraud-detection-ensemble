"""Build the README walkthrough GIF from the running dashboard (real replay, real model).

Requires the API running locally with a promoted model (``uvicorn api.main:app --port 8000``).
It resets the demo data, runs a fraud-enriched replay of held-out transactions, captures each
page with a headless browser, then a synthetic-drift replay for the monitoring frame.

    python scripts/make_demo_gif.py --url http://localhost:8000 --out docs/images/demo.gif
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_screenshots import capture_page, find_browser


def call(url: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"{}")


def font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def captioned(png: Path, caption: str, step: str, width: int) -> Image.Image:
    im = Image.open(png).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    bar = 46
    canvas = Image.new("RGB", (im.width, im.height + bar), (11, 11, 11))
    canvas.paste(im, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 11), step, fill=(134, 182, 239), font=font(19))
    draw.text(
        (16 + draw.textlength(step + "  ", font=font(19)), 11), caption, fill=(255, 255, 255), font=font(19)
    )
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="docs/images/demo.gif")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--frame-ms", type=int, default=2600)
    args = ap.parse_args()
    api = args.url.rstrip("/") + "/api/v1"
    browser = find_browser()

    call(f"{api}/admin/reset", "POST")
    call(f"{api}/replay/start", "POST", {"scenario": "enriched", "rate_tps": 60, "count": 1480, "seed": 7})
    plan: list[tuple[str, str, float]] = [
        ("overview", "Real held-out transactions replayed through the scoring API", 8),
        ("live", "Every transaction scored in about a millisecond, streamed live", 5),
        ("overview", "Calibrated fraud probabilities drive a versioned decision policy", 6),
        ("cases", "Analysts investigate alerts with TreeSHAP explanations and an audit trail", 0),
        ("performance", "Out-of-time evaluation with confidence intervals: XGBoost beat the ensemble", 0),
        ("policy", "Thresholds are business policy: what-if cost and review capacity", 0),
        ("monitoring", "Drift monitoring flags a synthetic distribution shift", 0),
    ]
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (page, caption, wait_s) in enumerate(plan, start=1):
            time.sleep(wait_s)
            target = page
            if page == "cases":
                alerts = call(f"{api}/transactions?alerts_only=true&action=decline&limit=50")["items"]
                if alerts:
                    target = f"cases/{max(alerts, key=lambda a: a['amount'])['transaction_id']}"
            if page == "monitoring":
                call(f"{api}/replay/stop", "POST")
                call(
                    f"{api}/replay/start",
                    "POST",
                    {"scenario": "synthetic_drift", "rate_tps": 150, "count": 2500, "seed": 3},
                )
                time.sleep(20)
            png = capture_page(browser, args.url, target, Path(tmp) / f"{i}.png", tmp + "/profile")
            frames.append(captioned(png, caption, f"{i}/{len(plan)}", args.width))
            print(f"frame {i}: {target}", flush=True)
        call(f"{api}/replay/stop", "POST")

    palette = [
        f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames
    ]
    durations = [args.frame_ms] * len(palette)
    durations[-1] = args.frame_ms * 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        args.out, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True
    )
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.2f} MB, {len(palette)} frames)")


if __name__ == "__main__":
    main()
