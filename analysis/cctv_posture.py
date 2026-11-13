"""Validate the posture signal on the CCTV incident dataset.

The dataset (simuletic/cctv-incident-dataset-fall-and-lying-down-detection) is a
collection of still CCTV frames with YOLO-format 17-keypoint annotations, labelled
laying (0) or standing (1). It is not a temporal dataset, so it cannot validate
fall events - but it can validate the single question the live badge and the
offline detector both depend on: does the image-space posture signal separate a
person on the floor from a person on their feet, across many scenes and camera
angles?

COCO 17 keypoints are mapped to the MediaPipe indices the posture signal reads
(shoulders 11/12, hips 23/24, ankles 27/28). Everything else is NaN, so the
signal sees exactly the joints it would see from a real capture.

Usage:
    python -m analysis.cctv_posture
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis import fall
from analysis.paths import CCTV_INCIDENT_LABELS as CCTV_DIR

# COCO 17 -> MediaPipe 33
COCO_TO_MP = {5: 11, 6: 12, 11: 23, 12: 24, 15: 27, 16: 28}


def _load_persons(label_path: Path) -> list[tuple[int, np.ndarray]]:
    """Return (class_id, landmarks (33,3)) for each annotated person."""
    persons = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        # bbox (cx, cy, w, h) then 17 x,y,vis triples
        kpts = parts[5:]
        lm = np.full((33, 3), np.nan)
        for i in range(17):
            x = float(kpts[3 * i])
            y = float(kpts[3 * i + 1])
            v = float(kpts[3 * i + 2])
            if v <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            if i in COCO_TO_MP:
                lm[COCO_TO_MP[i]] = [x, y, 0.0]
        persons.append((cls, lm))
    return persons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    d = args.dir
    if d is None:
        d = CCTV_DIR
    if d is None or not (d / "labels").exists():
        raise SystemExit("CCTV dataset not found. Download it first (see README).")

    img_dir = d / "images"
    label_dir = d / "labels"

    rows = []
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        # image may be png/jpg
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        # frame size: read image if possible, else assume 1024x1024
        from PIL import Image
        if img_path is not None:
            with Image.open(img_path) as im:
                w, h = im.size
        else:
            w, h = 1024, 1024

        for cls, lm in _load_persons(label_path):
            t = np.array([0.0])
            arr = lm.reshape(1, 33, 3)
            sig = fall.posture(t, arr, w, h)
            hr = float(sig["height_ratio"][0])
            ang = float(sig["angle"][0])
            valid = bool(sig["valid"][0])
            # same hysteresis thresholds as the live badge and the offline detector
            down = (ang > fall.DOWN_ANGLE) or (hr < fall.DOWN_HEIGHT)
            up = (ang < fall.UPRIGHT_ANGLE) and (hr > fall.UPRIGHT_HEIGHT)
            rows.append({
                "image": stem,
                "class": cls,
                "class_name": "laying" if cls == 0 else "standing",
                "height_ratio": round(hr, 3) if np.isfinite(hr) else None,
                "torso_angle_deg": round(ang, 1) if np.isfinite(ang) else None,
                "valid": valid,
                "predicted_down": down,
                "predicted_up": up,
            })

    # Report
    laying = [r for r in rows if r["class"] == 0]
    standing = [r for r in rows if r["class"] == 1]
    print(f"images: {len(list(label_dir.glob('*.txt')))}  persons: {len(rows)}  "
          f"(laying {len(laying)}, standing {len(standing)})")
    print(f"frame size: {w}x{h}")

    def _rate(subset, key):
        n = len(subset)
        k = sum(1 for r in subset if r[key])
        return k, n, (k / n * 100 if n else 0)

    ld, ln, lr = _rate(laying, "predicted_down")
    su, sn, sr = _rate(standing, "predicted_up")
    print(f"laying   -> predicted down:    {ld}/{ln} = {lr:.1f}%")
    print(f"standing -> predicted upright: {su}/{sn} = {sr:.1f}%")
    # confusion
    print(f"laying   -> predicted upright (false negative): {sum(1 for r in laying if r['predicted_up'])}")
    print(f"standing -> predicted down    (false positive): {sum(1 for r in standing if r['predicted_down'])}")
    # distributions
    def _stats(subset, key):
        vals = [r[key] for r in subset if r[key] is not None]
        if not vals:
            return "n/a"
        return f"median {np.median(vals):.2f}, range {min(vals):.2f}..{max(vals):.2f}"
    print("laying height_ratio:", _stats(laying, "height_ratio"))
    print("laying torso_angle:  ", _stats(laying, "torso_angle_deg"))
    print("standing height_ratio:", _stats(standing, "height_ratio"))
    print("standing torso_angle:  ", _stats(standing, "torso_angle_deg"))

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "summary": {
            "n_persons": len(rows),
            "laying_n": len(laying), "laying_down_rate": ld / ln if ln else None,
            "standing_n": len(standing), "standing_up_rate": su / sn if sn else None,
        }}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
