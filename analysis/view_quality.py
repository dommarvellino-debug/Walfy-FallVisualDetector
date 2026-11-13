"""Pose tracking quality per camera view, measured only where it is meaningful.

A low whole-video detection rate is not a failure: MCFD clips include long stretches
with nobody in frame. The useful question is whether the pose is tracked *inside the
labelled windows*, where a person is known to be present.

    python -m analysis.view_quality --subset chute01-chute08
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import labels
from analysis.paths import DERIVED


def rates_for(frames_file: Path, chute: int, cam: int, df) -> dict:
    present = []
    for line in frames_file.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        present.append(1 if r["lm"] else 0)
    present = np.array(present)

    mask = labels.frame_mask(df, chute, cam, len(present)).to_numpy()
    out = {"n": len(present), "overall": float(present.mean())}
    for name, value in (("fall", 1), ("nonfall", 0), ("ignore", -1)):
        sel = mask == value
        out[f"{name}_n"] = int(sel.sum())
        out[f"{name}_rate"] = float(present[sel].mean()) if sel.any() else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DERIVED / "mcfd")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    df = labels.load()
    rows = []
    for frames_file in sorted(args.root.rglob("frames.jsonl")):
        chute = int(frames_file.parent.parent.name.replace("chute", ""))
        cam = int(frames_file.parent.name.replace("cam", ""))
        r = rates_for(frames_file, chute, cam, df)
        r.update(chute=chute, cam=cam)
        rows.append(r)

    if not rows:
        print("no extracted frames found")
        return 1

    print(f"{'chute':>6}{'cam':>4}{'overall':>9}{'fall':>8}{'nonfall':>9}{'ignore':>8}{'fall_n':>8}")
    for r in rows:
        print(
            f"{r['chute']:>6}{r['cam']:>4}{r['overall']:>9.2f}{r['fall_rate']:>8.2f}"
            f"{r['nonfall_rate']:>9.2f}{r['ignore_rate']:>8.2f}{r['fall_n']:>8}"
        )

    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r["cam"]].append(r)

    print(f"\n{'cam':>4}{'n':>4}{'overall':>9}{'fall':>8}{'nonfall':>9}{'ignore':>8}")
    summary = []
    for cam in sorted(by_cam):
        v = by_cam[cam]
        entry = {"cam": cam, "n": len(v)}
        for key in ("overall", "fall_rate", "nonfall_rate", "ignore_rate"):
            vals = [x[key] for x in v if not np.isnan(x[key])]
            entry[key] = round(float(np.mean(vals)), 3) if vals else None
        summary.append(entry)
        print(
            f"{cam:>4}{len(v):>4}{entry['overall']:>9.2f}{entry['fall_rate']:>8.2f}"
            f"{entry['nonfall_rate']:>9.2f}{entry['ignore_rate']:>8.2f}"
        )

    if args.out:
        args.out.write_text(json.dumps({"videos": rows, "by_cam": summary}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
