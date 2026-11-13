"""Batch-extract MCFD videos to landmark streams.

    python -m analysis.batch_mcfd --chutes 1-8 --cams 1-8

Writes data/derived/mcfd/chuteNN/camM/{frames.jsonl,extract.json}.
Already-extracted videos are skipped, so re-running resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from analysis.extract import MCFD_VIDEOS, extract

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data/derived/mcfd"


def parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chutes", default="1-8")
    ap.add_argument("--cams", default="1-8")
    ap.add_argument("--model", default="lite", choices=["lite", "full", "heavy"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)

    chutes = parse_range(args.chutes)
    cams = parse_range(args.cams)
    total = len(chutes) * len(cams)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    done = skipped = failed = 0
    summaries = []

    for chute in chutes:
        for cam in cams:
            video = MCFD_VIDEOS / f"chute{chute:02d}" / f"cam{cam}.avi"
            out_dir = args.out / f"chute{chute:02d}" / f"cam{cam}"
            idx = done + skipped + failed + 1

            if (out_dir / "extract.json").exists():
                skipped += 1
                continue
            if not video.exists():
                failed += 1
                print(f"[{idx}/{total}] missing {video}", flush=True)
                continue

            try:
                summary = extract(video, out_dir, args.model, None, args.stride)
            except Exception as exc:  # keep the batch alive
                failed += 1
                print(f"[{idx}/{total}] FAILED {video.name}: {exc}", flush=True)
                continue

            done += 1
            summary["chute"] = chute
            summary["cam"] = cam
            summaries.append(summary)
            print(
                f"[{idx}/{total}] chute{chute:02d}/cam{cam} "
                f"det={summary['detection_rate']:.2f} "
                f"{summary['elapsed_s']:.1f}s "
                f"({summary['elapsed_s'] / max(summary['frames'], 1) * 1000:.0f} ms/frame)",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    report = {
        "model": args.model,
        "stride": args.stride,
        "chutes": chutes,
        "cams": cams,
        "extracted": done,
        "skipped": skipped,
        "failed": failed,
        "elapsed_s": round(elapsed, 1),
        "videos": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if summaries:
        rates = sorted(s["detection_rate"] for s in summaries)
        print(
            f"\ndone={done} skipped={skipped} failed={failed} in {elapsed / 60:.1f} min\n"
            f"detection rate: min={rates[0]:.2f} median={rates[len(rates) // 2]:.2f} max={rates[-1]:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
