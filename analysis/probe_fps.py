"""Probe whether MCFD's declared frame rate is real motion or duplicated frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MCFD = Path(
    r"D:\Walk tracker\data\raw\_kagglehub\datasets\soumicksarker"
    r"\multiple-cameras-fall-dataset\versions\2\dataset\dataset"
)


def probe(rel: str, n: int = 40) -> None:
    path = MCFD / rel
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev = None
    diffs = []
    while len(diffs) < n:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.abs(gray.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = gray
    cap.release()

    arr = np.array(diffs)
    zero = int((arr < 0.5).sum())
    print(f"{rel} declared_fps={fps:.0f} frames={len(arr)}")
    print(f"  diff mean={arr.mean():.3f} min={arr.min():.3f} max={arr.max():.3f}")
    print(f"  near-identical consecutive pairs: {zero}/{len(arr)}")
    print("  first 24 diffs: " + " ".join(f"{d:.2f}" for d in arr[:24]))


if __name__ == "__main__":
    for rel in sys.argv[1:] or ["chute01/cam1.avi", "chute12/cam3.avi"]:
        probe(rel)
