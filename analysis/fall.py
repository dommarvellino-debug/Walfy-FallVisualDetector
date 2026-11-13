"""Detect fall events from a landmark stream.

A fall is a *transition into a down state*, not a single frame. The event runs from
the moment the body goes down until the subject is upright again, because "someone
is on the floor" is the condition worth alerting on — a momentary dip of the hips is
not.

    python -m analysis.fall data/raw/<session>

Signals, both scale-invariant so the same thresholds work at any camera distance:

  height_ratio  hip height above the ankles, divided by torso length in pixels
  torso_angle   angle of the shoulder-hip axis from vertical, aspect-corrected

These are computed in image space on purpose. MediaPipe's world landmarks are not
reliably gravity-aligned when the body is horizontal: in chute01 a subject lying on
the floor still reads height_ratio 0.86 in world coordinates but 0.65 in image
coordinates, where standing reads 0.85. The image signal separates the states, the
world signal does not.

Gaps matter. Frames are only bridged for at most MAX_GAP_S, and a transition is
never allowed to span a longer dropout — otherwise a 6-second loss of tracking
becomes a smooth, invented collapse.

Not a medical device and not a fall alarm. It reports kinematic events; whether a
real person fell needs context this cannot see.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import medfilt, savgol_filter

from analysis.session import require_frames

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

MAX_GAP_S = 0.4

# Tuning lives here so it can be cited, changed and re-tested.
UPRIGHT_ANGLE = 35.0
DOWN_ANGLE = 50.0
UPRIGHT_HEIGHT = 0.55
DOWN_HEIGHT = 0.40
MIN_DOWN_S = 0.30
INITIATION_S = 2.0
RECOVERY_S = 0.6
MIN_UPRIGHT_BEFORE_S = 0.05

DEFAULT_ASPECT = 16 / 9


def read_frames(frames_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a frames.jsonl stream into (times, image landmarks, count)."""
    lines = frames_path.read_text(encoding="utf-8").splitlines()
    frames = [json.loads(line) for line in lines if line.strip()]

    n = len(frames)
    t = np.array([f["t"] for f in frames], dtype=float) / 1000.0
    lm = np.full((n, 33, 3), np.nan)
    for i, f in enumerate(frames):
        if f.get("lm"):
            lm[i] = np.array(f["lm"], dtype=float)[:, :3]
    return t, lm, n


def load_session(path: Path):
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    t, lm, n = read_frames(path / "frames.jsonl")
    return meta, t, lm, n


def frame_size(meta: dict, path: Path) -> tuple[int, int]:
    """Find the true pixel size of the frame.

    The aspect ratio matters: normalised landmark coordinates are anisotropic, so
    an angle computed without it is wrong by the aspect ratio. The capture app
    writes frame_width/frame_height into session.json; batch extraction writes
    extract.json next to the landmarks.
    """
    for key_w, key_h in (("frame_width", "frame_height"), ("width", "height")):
        if meta.get(key_w) and meta.get(key_h):
            return int(meta[key_w]), int(meta[key_h])
    for candidate in (path / "extract.json", path.parent / "extract.json"):
        if candidate.exists():
            m = json.loads(candidate.read_text(encoding="utf-8"))
            if m.get("width") and m.get("height"):
                return int(m["width"]), int(m["height"])
    return 0, 0


def fill_short_gaps(y: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate across dropouts no longer than max_gap frames.

    Anything longer stays NaN. Bridging a multi-second dropout would manufacture a
    smooth transition through frames the tracker never saw.
    """
    out = y.copy()
    n = len(out)
    idx = np.arange(n)

    for k in range(y.shape[1]):
        good = valid & np.isfinite(y[:, k])
        if good.sum() < 2:
            continue
        col = out[:, k].copy()

        # Enumerate missing runs and fill only the short ones.
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


def smooth(y: np.ndarray, dt: float) -> np.ndarray:
    """Median + Savitzky-Golay smoothing that tolerates NaN.

    Filtering is applied independently to each contiguous finite run, so a long
    dropout is never smoothed into a fake signal.
    """
    out = y.copy()
    k = max(3, int(round(0.10 / dt)))
    k = k + 1 if k % 2 == 0 else k
    w = max(5, int(round(0.20 / dt)))
    w = w + 1 if w % 2 == 0 else w

    n = len(out)
    i = 0
    while i < n:
        if not np.isfinite(out[i]).all():
            i += 1
            continue
        j = i
        while j < n and np.isfinite(out[j]).all():
            j += 1

        seg = out[i:j]
        kk = min(k, len(seg) // 2 * 2 - 1)
        if kk >= 3:
            seg = medfilt(seg, kernel_size=(kk, 1))
        ww = min(w, len(seg) // 2 * 2 - 1)
        if ww >= 5:
            seg = savgol_filter(seg, ww, 2, axis=0)
        out[i:j] = seg

        i = j
    return out


def posture(t: np.ndarray, lm: np.ndarray, width: int, height: int) -> dict:
    """Height ratio and torso angle in image space, aspect corrected."""
    if width and height:
        w, h = float(width), float(height)
    else:
        # Unknown frame size: assume a 16:9 webcam rather than leaving the angle
        # uncorrected (which would silently read every pose as upright).
        w, h = DEFAULT_ASPECT, 1.0
    sh = (lm[:, L_SHOULDER] + lm[:, R_SHOULDER]) / 2
    hp = (lm[:, L_HIP] + lm[:, R_HIP]) / 2
    an = (lm[:, L_ANKLE] + lm[:, R_ANKLE]) / 2

    valid = np.isfinite(sh).all(axis=1) & np.isfinite(hp).all(axis=1) & np.isfinite(an).all(axis=1)

    torso_px = (sh - hp) * np.array([w, h, h])
    torso_len = float(np.nanmedian(np.linalg.norm(torso_px, axis=1)))

    height_ratio = ((an[:, 1] - hp[:, 1]) * h) / max(torso_len, 1e-6)
    angle = np.degrees(np.arctan2(np.abs(torso_px[:, 0]), np.abs(torso_px[:, 1]) + 1e-9))

    return {"height_ratio": height_ratio, "angle": angle, "valid": valid, "torso_len_px": torso_len}


def step_states(h: np.ndarray, a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1 upright, -1 down, 0 unknown, via hysteresis on the continuous signals.

    Hysteresis instead of a median filter over the categorical state: a median
    filter delays every state edge by half its window, which is enough to place a
    fall event on the wrong side of a short label window. Entering the down state
    needs one threshold, leaving it needs a stricter one, so ordinary noise near a
    boundary cannot flip the state.
    """
    n = len(h)
    state = np.zeros(n, dtype=int)
    cur = 0

    for i in range(n):
        if not valid[i]:
            # Long dropout: the posture is unknown, and no event may span it.
            state[i] = 0
            cur = 0
            continue

        if cur == -1:
            if a[i] < UPRIGHT_ANGLE and h[i] > UPRIGHT_HEIGHT:
                cur = 1
        elif cur == 1:
            if a[i] > DOWN_ANGLE or h[i] < DOWN_HEIGHT:
                cur = -1
        else:
            if a[i] > DOWN_ANGLE or h[i] < DOWN_HEIGHT:
                cur = -1
            elif a[i] < UPRIGHT_ANGLE and h[i] > UPRIGHT_HEIGHT:
                cur = 1

        state[i] = cur

    return state


def drop_short_runs(state: np.ndarray, value: int, min_frames: int) -> np.ndarray:
    """Remove runs of `value` shorter than min_frames.

    A brief threshold crossing during the descent is not evidence that someone is
    on the floor. Only a sustained down state is treated as one.
    """
    out = state.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != value:
            i += 1
            continue
        j = i
        while j < n and out[j] == value:
            j += 1
        if (j - i) < min_frames:
            out[i:j] = 0
        i = j
    return out


def find_recovery(state: np.ndarray, start: int, rec: int) -> int | None:
    """First index after `start` where the subject has been upright long enough.

    Upright time accumulates rather than requiring one unbroken run: getting up
    passes through intermediate postures and may include a short pause, and a
    single unclean frame should not declare a fallen person unrecovered. Any
    return to the down state resets the credit.
    """
    n = len(state)
    up = 0
    k = start
    while k < n:
        if state[k] == 1:
            up += 1
            if up >= rec:
                return k
        elif state[k] == -1:
            up = 0
        k += 1
    return None


def detect(
    t: np.ndarray, h: np.ndarray, a: np.ndarray, state: np.ndarray, valid: np.ndarray, dt: float
) -> list[dict]:
    n = len(t)
    min_up = max(1, int(MIN_UPRIGHT_BEFORE_S / dt))
    init = max(1, int(INITIATION_S / dt))
    rec = max(1, int(RECOVERY_S / dt))

    events: list[dict] = []
    i = 0
    while i < n:
        if state[i] == 1:
            # Normal path: upright run followed by a down transition.
            up_from = i
            while i < n and state[i] == 1:
                i += 1
            if i - up_from < min_up:
                continue
            up_to = i

            j = up_to
            limit = min(n, up_to + init)
            while j < limit and state[j] != -1:
                if not valid[j]:
                    # A long dropout: the transition cannot be inferred across it.
                    j = limit
                    break
                j += 1
            if j >= limit or state[j] != -1:
                continue

            down_from = j
            while j < n and state[j] == -1:
                j += 1
            down_to = j

            rec_at = find_recovery(state, down_to, rec)
            seg = slice(down_from, max(down_to, down_from + 1))
            events.append(
                {
                    "t_upright_end_s": round(float(t[up_to - 1]), 3),
                    "t_down_s": round(float(t[down_from]), 3),
                    "t_down_end_s": round(float(t[min(down_to, n - 1)]), 3),
                    "down_duration_s": round(float(t[min(down_to, n - 1)] - t[down_from]), 3),
                    "recovered": rec_at is not None,
                    "t_recovered_s": round(float(t[rec_at]), 3) if rec_at is not None else None,
                    "min_height_ratio": round(float(np.min(h[seg])), 3),
                    "max_torso_angle_deg": round(float(np.max(a[seg])), 1),
                }
            )
            i = max(j, i + 1)

        else:
            i += 1

    return events


def analyse_frames(t: np.ndarray, lm: np.ndarray, width: int, height: int) -> dict:
    """Core detector over raw landmark arrays, with no session metadata needed.

    Returns the report plus the derived signals, so callers (the MCFD evaluator in
    particular) can audit exactly what the state machine saw rather than trusting a
    verdict.
    """
    n = len(t)
    sig = posture(t, lm, width, height)

    dt = float(np.median(np.diff(t))) if n > 2 else 1 / 30
    max_gap = max(1, int(MAX_GAP_S / dt))

    h = smooth(fill_short_gaps(sig["height_ratio"].reshape(-1, 1), sig["valid"], max_gap), dt)[:, 0]
    a = smooth(fill_short_gaps(sig["angle"].reshape(-1, 1), sig["valid"], max_gap), dt)[:, 0]

    # After gap filling, a frame is usable if both signals exist. Long dropouts stay
    # NaN, so the state machine refuses to infer a transition across them.
    valid = np.isfinite(h) & np.isfinite(a)

    state = step_states(h, a, valid)
    state = drop_short_runs(state, -1, max(1, int(MIN_DOWN_S / dt)))
    events = detect(t, h, a, state, valid, dt)

    return {
        "frames": n,
        "detection_rate": round(float(sig["valid"].mean()), 4),
        "torso_len_px": round(sig["torso_len_px"], 1),
        "frame_size": [width or None, height or None],
        "frame_size_assumed": not (width and height),
        "events": events,
        "n_events": len(events),
        "t": t,
        "height_ratio": h,
        "torso_angle": a,
        "valid": valid,
        "state": state,
    }


def analyse(path: Path) -> dict:
    meta, t, lm, n = load_session(path)
    width, height = frame_size(meta, path)
    out = analyse_frames(t, lm, width, height)

    if out["frame_size_assumed"]:
        print(
            f"warning: {path} has no frame size; assuming 16:9. "
            "Torso angle depends on the aspect ratio, so the state thresholds may "
            "be wrong for a non-16:9 camera.",
            file=sys.stderr,
        )

    report = {
        "session_id": meta.get("session_id"),
        "subject_id": meta.get("subject_id"),
        "frames": out["frames"],
        "detection_rate": out["detection_rate"],
        "torso_len_px": out["torso_len_px"],
        "frame_size": out["frame_size"],
        "frame_size_assumed": out["frame_size_assumed"],
        "thresholds": {
            "upright_angle_deg": UPRIGHT_ANGLE,
            "down_angle_deg": DOWN_ANGLE,
            "upright_height_ratio": UPRIGHT_HEIGHT,
            "down_height_ratio": DOWN_HEIGHT,
            "initiation_s": INITIATION_S,
            "recovery_s": RECOVERY_S,
            "min_down_s": MIN_DOWN_S,
            "min_upright_before_s": MIN_UPRIGHT_BEFORE_S,
            "max_gap_s": MAX_GAP_S,
        },
        "events": out["events"],
        "n_events": out["n_events"],
    }
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
    (args.json or (session / "fall.json")).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
