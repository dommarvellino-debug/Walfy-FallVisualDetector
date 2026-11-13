"""Extract a MediaPipe pose landmark stream from a video file.

Writes the frames.jsonl format described in docs/session-format.md so that
offline batch and live capture produce interchangeable artefacts.

    python -m analysis.extract clip.mp4 --out data/derived/clip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from analysis.paths import MODELS, MCFD_VIDEOS


def build_landmarker(model: str, running_mode: str):
    model_path = MODELS / f"pose_landmarker_{model}.task"
    if not model_path.exists():
        raise SystemExit(f"missing model file: {model_path}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=getattr(vision.RunningMode, running_mode.upper()),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def frame_record(t_ms: int, frame_idx: int, result) -> dict:
    if not result.pose_landmarks:
        return {"t": t_ms, "frame": frame_idx, "lm": None, "wlm": None}

    lm = [
        [round(p.x, 6), round(p.y, 6), round(p.z, 6), p.visibility, p.presence]
        for p in result.pose_landmarks[0]
    ]
    wlm = (
        [[round(p.x, 6), round(p.y, 6), round(p.z, 6)] for p in result.pose_world_landmarks[0]]
        if result.pose_world_landmarks
        else None
    )
    return {"t": t_ms, "frame": frame_idx, "lm": lm, "wlm": wlm}


def extract(
    video: Path,
    out_dir: Path,
    model: str = "lite",
    limit: int | None = None,
    stride: int = 1,
    start: int = 0,
) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")

    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    landmarker = build_landmarker(model, "video")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"

    processed = detected = 0
    read_idx = 0
    last_t = -1
    started = time.perf_counter()

    with frames_path.open("w", encoding="utf-8") as fh:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            src_idx = read_idx
            read_idx += 1
            if stride > 1 and src_idx % stride != 0:
                continue

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_ms = int(round(pos_ms)) if pos_ms and pos_ms > 0 else int(round(src_idx * 1000 / fps))
            if t_ms <= last_t:
                # MediaPipe rejects non-monotonic timestamps; duplicated PTS is
                # common in re-encoded video.
                t_ms = last_t + 1
            last_t = t_ms

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, t_ms)

            record = frame_record(t_ms, src_idx, result)
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

            processed += 1
            detected += 1 if record["lm"] else 0
            if limit and processed >= limit:
                break

    cap.release()
    landmarker.close()

    summary = {
        "video": str(video),
        "frames": processed,
        "declared_frames": declared,
        "width": width or None,
        "height": height or None,
        "detected": detected,
        "detection_rate": round(detected / processed, 4) if processed else 0.0,
        "fps": round(fps, 3),
        "stride": stride,
        "model": model,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "output": str(frames_path),
    }
    (out_dir / "extract.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="lite", choices=["lite", "full", "heavy"])
    ap.add_argument("--limit", type=int, default=None, help="stop after N processed frames")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth source frame")
    ap.add_argument("--start", type=int, default=0, help="first source frame to read")
    args = ap.parse_args(argv)

    summary = extract(args.video, args.out, args.model, args.limit, args.stride, args.start)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
