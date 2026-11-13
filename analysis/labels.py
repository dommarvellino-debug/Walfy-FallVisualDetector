"""Load and normalise MCFD temporal labels.

The label file has three defects that must be fixed before use:

1. ``cam == 55`` is a mis-keyed row that belongs to camera 3 (its window is missing
   from cam3 otherwise). Remapping restores cam3's full set rather than duplicating.
2. ``chute24`` has videos but no labels at all.
3. Windows are *sparse samples*, not a segmentation: large stretches of each video
   are unlabelled and must be treated as "ignore", never as negative. Only ~5-6% of
   frames carry a label, so naive whole-video accuracy is meaningless.

Overlapping windows merge when a mask is built (adjacent windows share a boundary
frame because both endpoints are inclusive).

Windows are inclusive of both endpoints, so the frame span is ``end - start + 1``.
"""

from __future__ import annotations

import pandas as pd

from analysis.paths import MCFD_LABELS

CAM_TYPO_FIX = {55: 3}


def load() -> pd.DataFrame:
    df = pd.read_csv(MCFD_LABELS)
    df["chute"] = df["chute"].astype(int)
    df["cam"] = df["cam"].astype(int)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["label"] = df["label"].astype(int)

    df["cam"] = df["cam"].replace(CAM_TYPO_FIX)
    df = df.drop_duplicates(subset=["chute", "cam", "start", "end", "label"])
    return df.sort_values(["chute", "cam", "start"]).reset_index(drop=True)


def windows(df: pd.DataFrame, chute: int, cam: int) -> list[tuple[int, int, int]]:
    """Return ``(start, end_exclusive, label)`` windows for one video."""
    sub = df[(df.chute == chute) & (df.cam == cam)]
    return [(int(r.start), int(r.end) + 1, int(r.label)) for r in sub.itertuples()]


def frame_mask(df: pd.DataFrame, chute: int, cam: int, n_frames: int) -> "pd.Series":
    """Per-frame label: 1 fall, 0 non-fall, -1 ignore (unlabelled)."""
    mask = [-1] * n_frames
    for start, end, label in windows(df, chute, cam):
        for i in range(max(0, start), min(n_frames, end)):
            mask[i] = label
    return pd.Series(mask, name="label")


def coverage(df: pd.DataFrame, chute: int, cam: int, n_frames: int) -> dict:
    mask = frame_mask(df, chute, cam, n_frames)
    n_fall = int((mask == 1).sum())
    n_non = int((mask == 0).sum())
    n_ign = int((mask == -1).sum())
    return {
        "frames": n_frames,
        "fall": n_fall,
        "nonfall": n_non,
        "ignore": n_ign,
        "labelled_frac": round((n_fall + n_non) / n_frames, 4) if n_frames else 0.0,
    }


if __name__ == "__main__":
    df = load()
    print(f"rows after cleaning: {len(df)}")
    print(f"cams: {sorted(df.cam.unique())}")
    print(f"chutes labelled: {df.chute.nunique()} (chute24 unlabelled by design)")
    print(f"label counts: {df.label.value_counts().to_dict()}")

    spans = (df.end - df.start + 1).unique()
    print(f"window span (frames): {sorted(spans.tolist())}")

    print("\nchute01/cam1 windows:", windows(df, 1, 1))
    print("coverage chute01/cam1 over 1562 frames:", coverage(df, 1, 1, 1562))
    print("coverage chute23/cam3 over a 4200-frame video:", coverage(df, 23, 3, 4200))
