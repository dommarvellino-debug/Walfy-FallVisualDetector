"""Compare the live JS posture tracker against the offline fall detector.

Runs tools/posture_parity.mjs on each MCFD landmark stream and measures how
closely the live, causal state agrees with the validated, non-causal one. The
point is not to claim the browser detector is validated - it is not - but to
publish an honest number for how much the two disagree, so the live badge never
silently contradicts fall.json.

Usage:
    python -m analysis.posture_agreement [--chutes 1-8] [--cams 1-8]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis import fall, labels
from analysis.paths import MCFD_VIDEOS, DERIVED


def _range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _state_int(s: str) -> int:
    return {"upright": 1, "down": -1, "unknown": 0}.get(s, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chutes", default="1-8")
    ap.add_argument("--cams", default="1-8")
    ap.add_argument("--node", default="node")
    args = ap.parse_args(argv)

    chutes = _range(args.chutes)
    cams = _range(args.cams)
    root = DERIVED / "mcfd"

    agree_frames = 0
    total_frames = 0
    agree_down = 0
    total_down = 0
    agree_up = 0
    total_up = 0
    events_js = 0
    events_py = 0
    matched = 0
    timing_diffs: list[float] = []
    n_videos = 0

    for ch in chutes:
        for cam in cams:
            d = root / f"chute{ch:02d}" / f"cam{cam}"
            if not (d / "frames.jsonl").exists():
                continue
            n_videos += 1
            # offline reference
            t, lm, n = fall.read_frames(d / "frames.jsonl")
            width, height = fall.frame_size({}, d)
            out = fall.analyse_frames(t, lm, width, height)
            py_state = out["state"]
            # live tracker
            proc = subprocess.run(
                [args.node, "tools/posture_parity.mjs", str(d)],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                print(f"node failed on {d}: {proc.stderr.strip()}", file=sys.stderr)
                continue
            js = json.loads(proc.stdout)
            js_state = np.array([_state_int(s) for s in js["states"]], dtype=int)
            n = min(len(py_state), len(js_state))
            py_state = py_state[:n]
            js_state = js_state[:n]

            total_frames += n
            agree_frames += int((py_state == js_state).sum())
            mask_down = py_state == -1
            mask_up = py_state == 1
            total_down += int(mask_down.sum())
            total_up += int(mask_up.sum())
            agree_down += int(((js_state == -1) & mask_down).sum())
            agree_up += int(((js_state == 1) & mask_up).sum())

            # event timing: match by t_down within 2 s
            js_events = js.get("events", [])
            py_events = out["events"]
            events_js += len(js_events)
            events_py += len(py_events)
            used = set()
            for je in js_events:
                for k, pe in enumerate(py_events):
                    if k in used:
                        continue
                    if abs(je["t_down_s"] - pe["t_down_s"]) <= 2.0:
                        used.add(k)
                        matched += 1
                        timing_diffs.append(je["t_down_s"] - pe["t_down_s"])
                        break

    print(f"videos: {n_videos}")
    print(f"frames: {total_frames}")
    if total_frames:
        print(f"overall state agreement: {agree_frames/total_frames*100:.2f}%")
    if total_down:
        print(f"down-state recall: {agree_down/total_down*100:.2f}% ({agree_down}/{total_down})")
    if total_up:
        print(f"upright-state recall: {agree_up/total_up*100:.2f}% ({agree_up}/{total_up})")
    print(f"events: js={events_js} py={events_py} matched={matched}")
    if timing_diffs:
        d = np.array(timing_diffs)
        print(f"event timing diff js-py: median {np.median(d):+.3f}s, range {d.min():+.3f}..{d.max():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
