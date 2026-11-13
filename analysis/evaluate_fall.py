"""Evaluate the fall detector against the MCFD labels.

    python -m analysis.evaluate_fall

Two label conventions are scored, because the dataset does not settle the question:

  impact_only        label 1 is the ~31-frame window around impact. The window that
                     follows it is labelled 0 while the subject is still on the floor.
  impact_plus_lying   the fall window is extended through that following label-0
                     window, i.e. "down" lasts until the subject is up again.

The first convention is what the file literally says. The second is what a fall
alarm actually needs. A detector cannot score well on both at once, so the point of
running this is to show the cost of each, not to pick a flattering number.

Only labelled frames count. Roughly 94-95% of frames are unlabelled and are excluded
from every rate; treating them as negatives would inflate specificity to
meaninglessness.

Several clips (all of chute05) begin already in the fall, with no labelled non-fall
window before impact. A transition detector cannot see a transition that happened
before the recording, so those are reported separately rather than buried in recall.

Temporal tolerance is reported as a curve. The labels mark impact on a sparse
sampling grid, so demanding frame-exact agreement would measure label jitter rather
than detection quality. Matching is asymmetric: an event may start up to
MATCH_EARLY_S before the labelled impact and still be the same fall, because a
useful alarm fires during the descent. False alarms are measured on quiet time only
— labelled non-fall windows at least QUIET_MARGIN_S from any fall — so an early
alarm or the post-impact lying phase is not miscounted as an error. Alert latency
is reported as event start minus labelled impact.

Outputs both a table and ``data/derived/fall_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis import fall, labels
from analysis.paths import DERIVED


CONVENTIONS = ("impact_only", "impact_plus_lying")
TOLERANCES_S = (0.0, 0.25, 0.5, 1.0)
BASELINE_LOOKBACK_S = 2.0
# An event that starts up to this long before the labelled impact is the same fall:
# labels mark impact, and a useful alarm fires during the descent. An event that
# ends before this window is a genuine false alarm.
MATCH_EARLY_S = 1.0
# A post-impact non-fall window is part of the lying phase even if the label file
# leaves a few frames between it and the fall window.
MERGE_GAP_S = 0.1
# A non-fall window this close to a fall is not "quiet" time: early warning and the
# lying phase both legitimately overlap it.
QUIET_MARGIN_S = 2.0


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching half-open spans."""
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def fall_spans(
    wins: list[tuple[int, int, int]], convention: str, merge_gap: int = 0
) -> list[tuple[int, int]]:
    """Frame spans that count as 'fall' under one label convention, episodes merged."""
    spans: list[tuple[int, int]] = []
    for idx, (start, end, lab) in enumerate(wins):
        if lab != 1:
            continue
        stop = end
        if convention == "impact_plus_lying":
            j = idx + 1
            # Allow a small label gap: chute06 ends its fall window and starts the
            # lying window six frames later. Physically the subject never got up.
            while j < len(wins) and wins[j][2] == 0 and wins[j][0] <= stop + merge_gap:
                stop = max(stop, wins[j][1])
                j += 1
        spans.append((start, stop))
    return merge_spans(spans)


def match_events(
    intervals: list[tuple[int, int]],
    spans: list[tuple[int, int]],
    early: int,
    late: int,
) -> dict[int, tuple[int, int]]:
    """Assign episodes to detected events, one event per episode.

    An event matches when its *start* lies in ``[span.start - early, span.end + late]``.
    Keying on the start rather than mere overlap matters twice: an event already
    running long before an episode is not evidence the detector saw that fall, and a
    single long event cannot be credited to two nearby episodes (chute06 has two
    episodes only 0.6s apart).
    """
    used: set[int] = set()
    matched: dict[int, tuple[int, int]] = {}
    for idx, (s, e) in enumerate(spans):
        cands = [
            (i, iv)
            for i, iv in enumerate(intervals)
            if i not in used and (s - early) <= iv[0] <= (e + late)
        ]
        if not cands:
            continue
        i, iv = min(cands, key=lambda c: abs(c[1][0] - s))
        used.add(i)
        matched[idx] = iv
    return matched


def mask_from_spans(spans: list[tuple[int, int]], n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[max(0, s) : min(n, e)] = True
    return m


def down_runs(state: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(state)
    while i < n:
        if state[i] != -1:
            i += 1
            continue
        j = i
        while j < n and state[j] == -1:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def t_to_index(t: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(t, value), 0, len(t) - 1))


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def gap_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Frames between two spans; 0 when they touch or overlap."""
    return max(0, b[0] - a[1], a[0] - b[1])


def observed_baseline(state: np.ndarray, start: int, lookback: int) -> bool:
    """Was the subject actually seen upright in the window before the episode?"""
    lo = max(0, start - lookback)
    return bool((state[lo:start] == 1).any())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived", type=Path, default=DERIVED / "mcfd")
    ap.add_argument("--json", type=Path, default=DERIVED / "fall_eval.json")
    args = ap.parse_args(argv)

    df = labels.load()
    videos = sorted(args.derived.glob("chute*/cam*/frames.jsonl"))
    if not videos:
        raise SystemExit(f"no derived videos under {args.derived}")

    per_video: list[dict] = []

    for frames_path in videos:
        chute = int(frames_path.parent.parent.name.replace("chute", ""))
        cam = int(frames_path.parent.name.replace("cam", ""))
        wins = labels.windows(df, chute, cam)
        if not wins:
            continue

        t, lm, n = fall.read_frames(frames_path)
        width, height = fall.frame_size({}, frames_path.parent)
        out = fall.analyse_frames(t, lm, width, height)

        dt = float(np.median(np.diff(t)))
        lookback = max(1, int(BASELINE_LOOKBACK_S / dt))
        state = out["state"]
        runs = down_runs(state)
        intervals = [
            (t_to_index(t, ev["t_down_s"]), t_to_index(t, ev["t_down_end_s"]) + 1)
            for ev in out["events"]
        ]

        rec = {
            "chute": chute,
            "cam": cam,
            "frames": n,
            "detection_rate": out["detection_rate"],
            "events": out["n_events"],
            "down_runs": len(runs),
            "intervals": intervals,
            "conventions": {},
        }

        early = max(1, int(round(MATCH_EARLY_S / dt)))
        merge_gap = max(0, int(round(MERGE_GAP_S / dt)))
        quiet_margin = max(0, int(round(QUIET_MARGIN_S / dt)))
        running = mask_from_spans(runs, n)
        all_labelled = mask_from_spans([(s, e) for s, e, _ in wins], n)
        unlabelled_mask = ~all_labelled
        unlabelled_frames = int(unlabelled_mask.sum())
        down_frames_unlabelled = int((running & unlabelled_mask).sum())

        for conv in CONVENTIONS:
            spans = fall_spans(wins, conv, merge_gap)
            nonfall_spans = [(s, e) for s, e, lab in wins if lab == 0]

            # Only windows far from every fall count as quiet time. A window next to
            # a fall will legitimately contain a down state (early alarm or lying
            # phase), so scoring it as a false positive punishes correct behaviour.
            quiet_spans = [
                (s, e)
                for s, e in nonfall_spans
                if all(gap_between((s, e), f) >= quiet_margin for f in spans)
            ]
            quiet_mask = mask_from_spans(quiet_spans, n)

            episodes = [
                {
                    "span": [s, e],
                    "baseline": observed_baseline(state, s, lookback),
                }
                for s, e in spans
            ]

            c: dict = {
                "fall_episodes": len(spans),
                "episodes_with_baseline": sum(1 for ep in episodes if ep["baseline"]),
                "nonfall_windows": len(nonfall_spans),
                "quiet_windows": len(quiet_spans),
                "quiet_windows_with_down": sum(
                    1 for w in quiet_spans if any(overlap(iv, w) for iv in intervals)
                ),
                "quiet_labelled_frames": int(quiet_mask.sum()),
                "down_frames_in_quiet": int((running & quiet_mask).sum()),
                "unlabelled_frames": unlabelled_frames,
                "down_frames_unlabelled": down_frames_unlabelled,
                "episodes": episodes,
                "recall": {},
                "spurious_events": {},
                "latency_s": [],
            }

            for tol_s in TOLERANCES_S:
                late = int(round(tol_s / dt))
                matched = match_events(intervals, spans, early, late)
                for idx, ep in enumerate(episodes):
                    ep[f"detected_{tol_s}"] = idx in matched
                detected_all = len(matched)
                detected_baseline = sum(
                    1 for idx in matched if episodes[idx]["baseline"]
                )
                c["recall"][str(tol_s)] = {
                    "all": detected_all,
                    "with_baseline": detected_baseline,
                }
                c["spurious_events"][str(tol_s)] = len(intervals) - len(matched)

            # Alert latency for episodes the detector did catch: negative means the
            # alarm fired during the descent, before the labelled impact.
            matched = match_events(intervals, spans, early, 0)
            for idx, iv in matched.items():
                s = spans[idx][0]
                if episodes[idx]["baseline"]:
                    c["latency_s"].append(round(float(t[iv[0]] - t[s]), 3))

            rec["conventions"][conv] = c

        per_video.append(rec)

    summary: dict = {"videos": len(per_video), "tolerances_s": list(TOLERANCES_S), "conventions": {}}

    for conv in CONVENTIONS:
        episodes = [ep for rec in per_video for ep in rec["conventions"][conv]["episodes"]]
        n_all = len(episodes)
        n_base = sum(1 for ep in episodes if ep["baseline"])
        videos = len(per_video)
        videos_with_baseline = sum(
            1 for rec in per_video if any(ep["baseline"] for ep in rec["conventions"][conv]["episodes"])
        )

        agg: dict = {
            "fall_episodes": n_all,
            "episodes_with_baseline": n_base,
            "videos": videos,
            "videos_with_baseline": videos_with_baseline,
            "nonfall_windows": sum(rec["conventions"][conv]["nonfall_windows"] for rec in per_video),
            "quiet_windows": sum(rec["conventions"][conv]["quiet_windows"] for rec in per_video),
            "quiet_windows_with_down": sum(
                rec["conventions"][conv]["quiet_windows_with_down"] for rec in per_video
            ),
            "quiet_labelled_frames": sum(
                rec["conventions"][conv]["quiet_labelled_frames"] for rec in per_video
            ),
            "down_frames_in_quiet": sum(
                rec["conventions"][conv]["down_frames_in_quiet"] for rec in per_video
            ),
            "unlabelled_frames": sum(
                rec["conventions"][conv]["unlabelled_frames"] for rec in per_video
            ),
            "down_frames_unlabelled": sum(
                rec["conventions"][conv]["down_frames_unlabelled"] for rec in per_video
            ),
            "recall": {},
            "spurious_events": {},
        }

        for tol_s in TOLERANCES_S:
            key = str(tol_s)
            agg["recall"][key] = {
                "all": sum(rec["conventions"][conv]["recall"][key]["all"] for rec in per_video),
                "with_baseline": sum(
                    rec["conventions"][conv]["recall"][key]["with_baseline"] for rec in per_video
                ),
            }
            agg["spurious_events"][key] = sum(
                rec["conventions"][conv]["spurious_events"][key] for rec in per_video
            )

        latencies = sorted(
            lat for rec in per_video for lat in rec["conventions"][conv]["latency_s"]
        )
        agg["latency_s"] = {
            "n": len(latencies),
            "median": round(float(np.median(latencies)), 3) if latencies else None,
            "min": latencies[0] if latencies else None,
            "max": latencies[-1] if latencies else None,
            "n_early": sum(1 for lat in latencies if lat < 0),
        }

        agg["quiet_window_fp_rate"] = (
            round(agg["quiet_windows_with_down"] / agg["quiet_windows"], 3)
            if agg["quiet_windows"]
            else None
        )
        agg["quiet_frame_fp_rate"] = (
            round(agg["down_frames_in_quiet"] / agg["quiet_labelled_frames"], 3)
            if agg["quiet_labelled_frames"]
            else None
        )
        agg["down_unlabelled_rate"] = round(
            agg["down_frames_unlabelled"] / max(agg["unlabelled_frames"], 1), 3
        )
        summary["conventions"][conv] = agg

    # Per-camera view at the mid tolerance.
    mid = 0.5
    by_cam: dict[int, dict] = {}
    for rec in per_video:
        c = by_cam.setdefault(
            rec["cam"],
            {"videos": 0, "episodes": 0, "episodes_with_baseline": 0, "detected_baseline": 0, "detection_rate": 0.0},
        )
        conv = rec["conventions"]["impact_plus_lying"]
        c["videos"] += 1
        c["episodes"] += conv["fall_episodes"]
        c["episodes_with_baseline"] += conv["episodes_with_baseline"]
        c["detected_baseline"] += conv["recall"][str(mid)]["with_baseline"]
        c["detection_rate"] += rec["detection_rate"]
    for c in by_cam.values():
        c["detection_rate"] = round(c["detection_rate"] / c["videos"], 3)
        c["recall_with_baseline"] = round(
            c["detected_baseline"] / max(c["episodes_with_baseline"], 1), 3
        )
    summary["by_cam"] = {str(k): by_cam[k] for k in sorted(by_cam)}

    report = {"summary": summary, "videos": per_video}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = summary
    print(f"videos scored: {s['videos']}  (chute24 has no labels and is excluded)\n")

    primary = "impact_plus_lying"
    c = s["conventions"][primary]
    print(
        f"{primary}: {c['fall_episodes']} episodes from {c['videos']} videos, "
        f"{c['episodes_with_baseline']} with an observed upright baseline"
    )
    print(f"  {'tolerance':>10}{'recall (all)':>16}{'recall (baseline)':>20}{'spurious':>10}")
    for tol_s in TOLERANCES_S:
        key = str(tol_s)
        r = c["recall"][key]
        print(
            f"  {tol_s:>9.2f}s"
            f"{r['all']:>10}/{c['fall_episodes']:<5}"
            f"{r['with_baseline']:>14}/{c['episodes_with_baseline']:<5}"
            f"{c['spurious_events'][key]:>10}"
        )

    lat = c["latency_s"]
    if lat["n"]:
        print(
            f"  alert latency (n={lat['n']}): median {lat['median']:+.2f}s, "
            f"range {lat['min']:+.2f}..{lat['max']:+.2f}s, "
            f"{lat['n_early']} before the labelled impact"
        )
    if c["quiet_windows"]:
        print(
            f"  quiet-window FP rate   {c['quiet_window_fp_rate']:.3f} "
            f"({c['quiet_windows_with_down']}/{c['quiet_windows']})"
        )
        print(
            f"  quiet-frame FP rate    {c['quiet_frame_fp_rate']:.3f} "
            f"({c['down_frames_in_quiet']}/{c['quiet_labelled_frames']} labelled frames)"
        )
    else:
        print(
            f"  false alarms            NOT MEASURABLE: 0 of {c['nonfall_windows']} "
            f"labelled non-fall windows are >={QUIET_MARGIN_S:g}s from a fall"
        )
        print(
            "                          (MCFD labels only the cushion around each fall, "
            "so no ordinary activity is ever labelled negative)"
        )
    print(
        f"  diagnostic only         down state on {c['down_unlabelled_rate']:.3f} "
        f"of unlabelled frames ({c['down_frames_unlabelled']}/{c['unlabelled_frames']})"
    )

    other = "impact_only"
    oc = s["conventions"][other]
    same = all(
        oc["recall"][str(t)]["all"] == c["recall"][str(t)]["all"] for t in TOLERANCES_S
    )
    print(
        f"\n{other}: "
        + ("identical counts" if same else "differs")
        + " - matching keys on the event start, so extending an episode through the"
    )
    print(
        "lying phase changes no recall; it would only change a false-alarm measure,"
    )
    print("which these labels cannot supply either.")

    print(f"\nper camera ({primary}, tolerance {mid}s)")
    print(f"{'cam':>4}{'videos':>8}{'episodes':>10}{'w/ baseline':>13}{'caught':>8}{'recall':>9}{'pose%':>8}")
    for cam, c in summary["by_cam"].items():
        print(
            f"{cam:>4}{c['videos']:>8}{c['episodes']:>10}{c['episodes_with_baseline']:>13}"
            f"{c['detected_baseline']:>8}{c['recall_with_baseline']:>9.2f}{c['detection_rate']*100:>7.0f}%"
        )

    print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
