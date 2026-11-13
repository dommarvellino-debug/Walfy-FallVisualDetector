"""Turn a captured TUG session into measurements.

    python -m analysis.tug data/raw/S01-tug-2026-11-13T18-00-00

Primary output is duration — the actual TUG outcome and the only number here that
does not depend on camera calibration. Step metrics are scale-free as well.
Metres are not attempted: this session format carries no px->m calibration.

Analysis runs on the longest contiguous stretch where the pose is present. Gaps are
never bridged across the whole clip, because linear interpolation across a dropout
invents a perfectly smooth walk that never happened.

Not a medical device. `duration_s` is a camera-derived estimate of what an operator
with a stopwatch would write down, nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, medfilt, savgol_filter

from analysis.session import require_frames

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

MIN_RUN_FRAMES = 30
# Same limit as the fall detector: a dropout longer than this is not bridged. Linear
# interpolation across a multi-second loss invents a smooth walk that never happened.
MAX_GAP_S = 0.4


def load_session(path: Path):
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    lines = (path / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    frames = [json.loads(line) for line in lines if line.strip()]

    n = len(frames)
    t = np.array([f["t"] for f in frames], dtype=float) / 1000.0
    lm = np.full((n, 33, 3), np.nan)
    vis = np.zeros((n, 33))
    wlm = np.full((n, 33, 3), np.nan)

    for i, f in enumerate(frames):
        if f.get("lm"):
            lm[i] = np.array(f["lm"], dtype=float)[:, :3]
            vis[i] = [p[3] if p[3] is not None else 1.0 for p in f["lm"]]
        if f.get("wlm"):
            wlm[i] = np.array(f["wlm"], dtype=float)[:, :3]

    return meta, t, lm, vis, wlm, n


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(mask) - start > best[1] - best[0]:
        best = (start, len(mask))
    return best


def fill_short_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate across dropouts no longer than max_gap frames; leave longer ones NaN."""
    out = y.copy()
    n = len(y)
    idx = np.arange(n)

    for k in range(y.shape[1]):
        good = np.isfinite(y[:, k])
        if good.sum() < 2:
            continue
        col = out[:, k].copy()
        i = 0
        while i < n:
            if good[i]:
                i += 1
                continue
            j = i
            while j < n and not good[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                col[i:j] = np.interp(idx[i:j], idx[good], y[good, k])
            i = j
        out[:, k] = col
    return out


def robust_smooth(y: np.ndarray, dt: float) -> np.ndarray:
    """Median filter then Savitzky-Golay.

    MediaPipe world landmarks are noisy at these subject scales; unfiltered ankle
    positions imply foot speeds of several metres per second, which no walker
    produces. The median stage removes tracker flips, the polynomial stage smooths
    the remainder without flattening the step cycle.
    """
    k = max(3, int(round(0.10 / dt)))
    if k % 2 == 0:
        k += 1
    k = min(k, len(y) // 2 * 2 - 1)
    if k >= 3:
        y = medfilt(y, kernel_size=(k, 1))

    w = max(5, int(round(0.20 / dt)))
    if w % 2 == 0:
        w += 1
    w = min(w, len(y) // 2 * 2 - 1)
    if w >= 5:
        y = savgol_filter(y, w, 2, axis=0)
    return y


def jitter_speed(t: np.ndarray, pts: np.ndarray, plane: list[int]) -> float:
    """Median ankle speed in m/s across consecutive observed frames only.

    Speed is only defined between two frames where the ankle was actually seen; a
    pair spanning a dropout is not a fast foot, it is a missing observation.
    """
    if len(t) < 3:
        return 0.0
    dt = float(np.median(np.diff(t)))
    a = pts[:, L_ANKLE][:, plane]
    good = np.isfinite(a).all(axis=1)
    both = good[:-1] & good[1:]
    if not both.any():
        return 0.0
    v = np.linalg.norm(np.diff(a, axis=0), axis=1) / dt
    return float(np.median(v[both]))


def step_metrics(t: np.ndarray, pts: np.ndarray, use_world: bool) -> dict:
    """Count steps from each foot's maximum forward displacement.

    A step for a given foot is its forward peak relative to the body, the standard
    gait-event definition. Using position relative to the hip midpoint cancels
    whole-body translation, so this works whether the subject walks toward, away
    from, or across the camera.

    Cadence is sanity-checked: an implied step time outside 0.25-1.5 s is reported
    as unreliable rather than passed off as a measurement.
    """
    # Pick the ground plane. In world coordinates y is vertical, so the horizontal
    # plane is (x, z). In normalised image coordinates the plane is (x, y).
    plane = [0, 2] if use_world else [0, 1]
    a_l = pts[:, L_ANKLE][:, plane]
    a_r = pts[:, R_ANKLE][:, plane]
    hip = (pts[:, L_HIP][:, plane] + pts[:, R_HIP][:, plane]) / 2

    dt = float(np.median(np.diff(t)))
    max_gap = max(1, int(round(MAX_GAP_S / dt)))
    a_l = fill_short_gaps(a_l, max_gap)
    a_r = fill_short_gaps(a_r, max_gap)
    hip = fill_short_gaps(hip, max_gap)

    # Count steps only inside one genuinely continuous stretch. Bridging a longer
    # dropout would let a peak detector invent a step cycle across it.
    ok = np.isfinite(a_l).all(axis=1) & np.isfinite(a_r).all(axis=1) & np.isfinite(hip).all(axis=1)
    lo, hi = longest_run(ok)
    if hi - lo < MIN_RUN_FRAMES:
        return {
            "error": "insufficient continuous ankle data; no stretch long enough to count steps"
        }

    # Jitter is measured on the original stream, using only pairs of observed frames.
    raw_jitter = jitter_speed(t, pts, plane)

    a_l, a_r, hip, t = a_l[lo:hi], a_r[lo:hi], hip[lo:hi], t[lo:hi]

    a_l = robust_smooth(a_l, dt)
    a_r = robust_smooth(a_r, dt)
    hip = robust_smooth(hip, dt)

    sep = a_l - a_r
    try:
        _, _, vt = np.linalg.svd(sep - sep.mean(axis=0), full_matrices=False)
    except np.linalg.LinAlgError:
        return {"error": "degenerate foot motion", "steps": 0}
    axis = vt[0]

    p_l = (a_l - hip) @ axis
    p_r = (a_r - hip) @ axis
    spread = float(np.std(np.concatenate([p_l, p_r])))
    if spread < 1e-6:
        return {"steps": 0, "note": "feet never separated; no walking detected"}

    min_gap = max(1, int(0.25 / dt))
    prominence = 0.4 * spread
    peaks_l, _ = find_peaks(p_l, prominence=prominence, distance=min_gap)
    peaks_r, _ = find_peaks(p_r, prominence=prominence, distance=min_gap)

    events = np.sort(np.concatenate([peaks_l, peaks_r]))
    result = {
        "steps": int(len(events)),
        "median_ankle_speed_mps_raw": round(raw_jitter, 2) if use_world else None,
    }

    if len(events) < 2:
        result["note"] = "no step cycle detected"
        return result

    intervals = np.diff(t[events])
    intervals = intervals[(intervals > 0.15) & (intervals < 3.0)]
    if not len(intervals):
        result["note"] = "step intervals implausible"
        return result

    mean_step = float(np.mean(intervals))
    result["step_time_mean_s"] = round(mean_step, 3)
    result["step_time_cv"] = round(float(np.std(intervals) / mean_step), 3)
    result["cadence_steps_per_min"] = round(60.0 / mean_step, 1)

    if not 0.25 <= mean_step <= 1.5:
        result["reliability"] = (
            f"implausible step time ({mean_step:.2f}s); treat step metrics as unreliable"
        )
    if use_world and raw_jitter > 3.0:
        result["reliability"] = (
            f"high landmark jitter ({raw_jitter:.1f} m/s ankle speed before filtering); "
            "step metrics are approximate"
        )
    return result


def torso_metrics(pts: np.ndarray) -> dict:
    """Torso angle from vertical. World landmarks only; normalised image
    coordinates are anisotropic, so an angle there depends on aspect ratio."""
    if not np.isfinite(pts[:, L_SHOULDER, 0]).any():
        return {"torso_angle": None, "note": "needs world landmarks"}

    sh = (pts[:, L_SHOULDER] + pts[:, R_SHOULDER]) / 2
    hp = (pts[:, L_HIP] + pts[:, R_HIP]) / 2
    v = sh - hp
    ang = np.degrees(np.arctan2(np.hypot(v[:, 0], v[:, 2]), np.abs(v[:, 1]) + 1e-9))
    return {
        "torso_angle_max_deg": round(float(np.nanmax(ang)), 1),
        "torso_angle_median_deg": round(float(np.nanmedian(ang)), 1),
        "hip_vertical_range": round(float(np.nanmax(hp[:, 1]) - np.nanmin(hp[:, 1])), 4),
    }


def analyse(path: Path) -> dict:
    meta, t, lm, vis, wlm, n = load_session(path)
    duration = meta.get("duration_ms", (t[-1] - t[0]) * 1000) / 1000.0

    valid_lm = np.isfinite(lm[:, L_ANKLE, 0])
    valid_w = np.isfinite(wlm[:, L_ANKLE, 0])
    a, b = longest_run(valid_lm | valid_w)
    span = b - a

    report = {
        "session_id": meta.get("session_id"),
        "subject_id": meta.get("subject_id"),
        "view": meta.get("view"),
        "frames": n,
        "detection_rate": round(float((valid_lm | valid_w).mean()), 4),
        "duration_s": round(float(duration), 3),
        "stopwatch_s": meta.get("stopwatch_s"),
        "fps_effective": round(float(1.0 / np.median(np.diff(t))), 2) if n > 2 else None,
    }

    if span < MIN_RUN_FRAMES:
        report["error"] = "no usable stretch of pose data"
        report["usable_span_s"] = round(float((t[b - 1] - t[a]) if b > a else 0.0), 3)
        return report

    world_cov = float(valid_w[a:b].mean())
    use_world = world_cov > 0.95
    pts = (wlm if use_world else lm)[a:b]
    seg_t = t[a:b]

    report.update(
        {
            "coordinate_source": "world" if use_world else "normalised",
            "usable_span_s": round(float(seg_t[-1] - seg_t[0]), 3),
            "usable_fraction": round(float(span / n), 3),
        }
    )

    if report["usable_fraction"] < 0.5:
        report["warning"] = (
            "less than half the recording has usable pose data; "
            "duration is fine but step metrics may be incomplete"
        )

    report["steps"] = step_metrics(seg_t, pts, use_world)
    report["torso"] = torso_metrics(pts)

    if meta.get("stopwatch_s"):
        report["timing_error_s"] = round(report["duration_s"] - float(meta["stopwatch_s"]), 3)

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    session = require_frames(args.session)
    report = analyse(session)
    text = json.dumps(report, indent=2)
    print(text)

    out = args.json or (session / "report.json")
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
