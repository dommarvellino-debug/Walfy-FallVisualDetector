"""Locate a session on disk before analysing it.

A session can arrive two ways:

  data/raw/<session_id>/session.json + frames.jsonl     folder picker, or unzipped
  <session_id>_session.json                             single-file browser download

Both resolve to the directory, so the analysers never care which happened. What
they do care about is ``frames.jsonl``: ``session.json`` is metadata only, and
without the landmark stream there is nothing to measure.

This module is also the one-command entry point:

    python -m analysis.session data/raw/<session_id>

It runs the TUG analysis and the fall scan over the same landmark stream and
prints a short summary; the same two JSON files the individual tools write are
saved beside the session. Pass ``--json`` to print the full reports instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def require_frames(path: Path) -> Path:
    """Return the session directory, or exit with a message a user can act on."""
    path = Path(path)
    if path.is_file():
        # Accept a lone session.json, including the browser's renamed download.
        meta = path
        path = path.parent
    else:
        meta = path / "session.json"

    if not meta.exists():
        raise SystemExit(
            f"no session.json in {path}\n"
            "  Point this at a session directory, or at the saved session.json file."
        )

    if not (path / "frames.jsonl").exists():
        rename = (
            ""
            if meta.name == "session.json"
            else (
                f"\n  {meta.name} is a lone download: rename it to session.json and put\n"
                "  it in a folder beside frames.jsonl."
            )
        )
        raise SystemExit(
            f"no frames.jsonl in {path}\n"
            "  session.json holds metadata only; the landmark stream is what gets analysed.\n"
            "  If the browser saved a .zip, unpack it here first. If only session.json\n"
            f"  was downloaded, the recording cannot be recovered - record it again.{rename}"
        )

    if meta.name != "session.json":
        raise SystemExit(
            f"found {meta.name}; the analysers read session.json by that exact name.\n"
            f"  Rename it to {path / 'session.json'}"
        )

    return path


# --------------------------------------------------------------------------
# one-command runner
# --------------------------------------------------------------------------


def _secs(value: float | None) -> str:
    return "?" if value is None else f"{value:.3f} s"


def _step_line(steps: dict) -> str:
    if steps.get("error"):
        return f"steps           not measurable ({steps['error']})"
    if "cadence_steps_per_min" not in steps:
        note = steps.get("note", "no step cycle found")
        return f"steps           {steps.get('steps', 0)} ({note})"
    return (
        f"steps           {steps['steps']}  "
        f"({steps['cadence_steps_per_min']:.1f} steps/min, "
        f"step time {steps['step_time_mean_s']:.3f} s, "
        f"CV {steps['step_time_cv']:.2f})"
    )


def _fall_lines(falls: dict) -> list[str]:
    events = falls.get("events") or []
    if not events:
        return ["  events          none: no transition into a down state was observed"]
    lines = [f"  events          {len(events)}"]
    for ev in events:
        lines.append("")
        lines.append(
            f"    down {ev['t_down_s']:.3f} s -> {ev['t_down_end_s']:.3f} s "
            f"({ev['down_duration_s']:.3f} s)"
        )
        if ev["recovered"]:
            lines.append(f"    upright again at {ev['t_recovered_s']:.3f} s")
        else:
            lines.append("    not seen upright again before the clip ends")
        lines.append(
            f"    min height ratio {ev['min_height_ratio']:.2f}, "
            f"max torso angle {ev['max_torso_angle_deg']:.1f} deg"
        )
    return lines


def summarise(report: dict, falls: dict) -> str:
    """Human summary of both analyses. ASCII only: the console mangles the rest."""
    out: list[str] = []
    out.append(f"{report.get('session_id') or 'unnamed session'}  ({report.get('subject_id')})")

    size = falls.get("frame_size")
    frame = f"{size[0]}x{size[1]}" if size else "size unknown"
    out.append(
        f"  {frame}, {report['frames']} frames, "
        f"{report['detection_rate'] * 100:.1f}% pose, "
        f"{report['duration_s']:.3f} s at {report.get('fps_effective') or '?'} fps"
    )
    out.append("")

    out.append("TUG")
    out.append(f"  duration        {_secs(report['duration_s'])}")
    if report.get("stopwatch_s"):
        out.append(
            f"  stopwatch       {_secs(float(report['stopwatch_s']))}  "
            f"(error {report['timing_error_s']:+.3f} s)"
        )
    else:
        out.append("  stopwatch       not entered, so timing cannot be checked")

    if report.get("error"):
        out.append(f"  unusable        {report['error']}")
    else:
        if report.get("usable_span_s") is not None:
            out.append(
                f"  usable pose     {_secs(report['usable_span_s'])} "
                f"({(report.get('usable_fraction') or 0) * 100:.0f}% of the recording, "
                f"{report.get('coordinate_source')} landmarks)"
            )
        out.append(f"  {_step_line(report.get('steps') or {})}")
        torso = report.get("torso") or {}
        if torso.get("torso_angle_median_deg") is not None:
            out.append(
                f"  torso           median {torso['torso_angle_median_deg']:.1f} deg, "
                f"max {torso['torso_angle_max_deg']:.1f} deg"
            )
    if report.get("warning"):
        out.append(f"  warning         {report['warning']}")
    out.append("")

    out.append("FALL")
    out.extend(_fall_lines(falls))
    if falls.get("frame_size_assumed"):
        out.append("")
        out.append(
            "  frame size unknown, assumed 16:9 - torso angle depends on the aspect"
        )
        out.append("  ratio, so the state thresholds may not suit this camera")
    out.append("")

    out.append("  Not a medical device. Duration estimates what a stopwatch would record;")
    out.append("  fall events are kinematics, not a clinical judgement.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m analysis.session",
        description="Run the TUG analysis and the fall scan on one session.",
    )
    ap.add_argument("session", type=Path)
    ap.add_argument("--json", action="store_true", help="print the full reports as JSON")
    ap.add_argument(
        "--no-write", action="store_true", help="do not write report.json / fall.json"
    )
    args = ap.parse_args(argv)

    session = require_frames(args.session)

    # Deferred: both analysers import require_frames from this module.
    from analysis import fall, tug

    report = tug.analyse(session)
    falls = fall.analyse(session)

    if not args.no_write:
        (session / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (session / "fall.json").write_text(json.dumps(falls, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({"tug": report, "fall": falls}, indent=2))
    else:
        print(summarise(report, falls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
