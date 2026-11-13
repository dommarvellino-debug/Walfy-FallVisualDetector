"""Print landmark-derived signals from a frames.jsonl for eyeballing.

    python -m analysis.quicklook data/derived/clip/frames.jsonl --every 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, hips, shoulder_mid = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ts.append(r["t"])
            if not r["lm"]:
                hips.append([np.nan, np.nan])
                shoulder_mid.append([np.nan, np.nan])
                continue
            lm = np.array(r["lm"])
            hips.append(((lm[L_HIP, :2] + lm[R_HIP, :2]) / 2).tolist())
            shoulder_mid.append(((lm[L_SHOULDER, :2] + lm[R_SHOULDER, :2]) / 2).tolist())
    return np.array(ts), np.array(hips), np.array(shoulder_mid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()

    ts, hips, sh = load(args.frames)

    # Normalised coords are anisotropic; angles need pixel space.
    width, height = 720, 480
    meta = args.frames.parent / "extract.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if "width" in m and "height" in m:
            width, height = m["width"], m["height"]

    t0 = ts[0]
    print(f"frames={len(ts)} span={(ts[-1] - t0) / 1000:.2f}s missing={int(np.isnan(hips[:, 0]).sum())}")
    print(f"{'t(s)':>6} {'hip_y':>7} {'hip_x':>7} {'torso_deg':>9}")
    for i in range(0, len(ts), args.every):
        if np.isnan(hips[i, 0]):
            print(f"{(ts[i] - t0) / 1000:6.2f} {'--':>7} {'--':>7} {'--':>9}")
            continue
        d = (sh[i] - hips[i]) * np.array([width, height])
        torso = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-6)))
        print(f"{(ts[i] - t0) / 1000:6.2f} {hips[i, 1]:7.3f} {hips[i, 0]:7.3f} {torso:9.1f}")

    ok = ~np.isnan(hips[:, 1])
    if ok.sum() > 2:
        print(f"\nhip_y range: {np.nanmin(hips[:, 1]):.3f} -> {np.nanmax(hips[:, 1]):.3f}"
              f" (spread {np.nanmax(hips[:, 1]) - np.nanmin(hips[:, 1]):.3f} of frame height)")


if __name__ == "__main__":
    main()
