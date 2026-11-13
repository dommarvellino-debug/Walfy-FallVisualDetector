"""Inventory the Multiple Cameras Fall Dataset and its label file.

    python -m analysis.inventory
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MCFD = (
    ROOT
    / "data/raw/_kagglehub/datasets/soumicksarker/multiple-cameras-fall-dataset/versions/2"
)


def main() -> None:
    df = pd.read_csv(MCFD / "data_tuple3.csv")
    video_root = MCFD / "dataset/dataset"

    print(f"label rows : {len(df)}")
    print(f"chutes     : {sorted(df.chute.unique())}")
    print(f"cams       : {sorted(df.cam.unique())}")
    print(f"labels     : {df.label.value_counts().to_dict()}")

    per = df.groupby(["chute", "cam"]).size().unstack(fill_value=0)
    print("\nrows per chute/cam (truncated):")
    print(per.head(8).to_string())

    print("\nchute01 rows:")
    print(df[df.chute == 1].to_string(index=False))

    folders = sorted(p.name for p in video_root.iterdir() if p.is_dir())
    missing = sorted(set(folders) - {f"chute{int(c):02d}" for c in df.chute.unique()})
    print(f"\nfolders: {len(folders)}, no labels in csv: {missing}")

    probe = video_root / "chute01/cam1.avi"
    cap = cv2.VideoCapture(str(probe))
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    print(f"\n{probe.relative_to(video_root)}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
