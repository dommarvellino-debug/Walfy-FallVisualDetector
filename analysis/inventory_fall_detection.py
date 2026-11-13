"""Inventory the darshantiwari17/fall-detection Kaggle dataset.

    python -m analysis.inventory_fall_detection

This is an IMU-based fall dataset (accelerometer + gyroscope features), complementary
to the video-based MCFD. It cannot be fed through the pose-landmark pipeline directly,
but it provides labeled fall/non-fall segments for cross-validating thresholds and
researching IMU-based fall detection.

Sources: SisFall, UCI_POSTURAL, UCI_HAR
Labels: NORMAL, FALL, STATIONARY, LYING
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.paths import FALL_DETECTION_ROOT


def main() -> None:
    csv = FALL_DETECTION_ROOT / "combined_features.csv"
    if not csv.exists():
        raise SystemExit(
            f"dataset not found at {csv}\n"
            "  Run: python -c \"import kagglehub; kagglehub.dataset_download('darshantiwari17/fall-detection')\"\n"
            "  then copy combined_features.csv into data/raw/_kagglehub/datasets/darshantiwari17/fall-detection/versions/1/"
        )

    df = pd.read_csv(csv)
    print(f"rows : {len(df):,}")
    print(f"cols : {list(df.columns)}")
    print(f"\nlabel distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\ndataset sources:\n{df['dataset'].value_counts().to_string()}")
    print(f"\nunique subjects: {df['subject_id'].nunique()}")
    print(f"subject sample: {sorted(df['subject_id'].unique())[:10]}")

    # Cross-tab: label by source
    ct = pd.crosstab(df["label"], df["dataset"])
    print(f"\nlabel x dataset:\n{ct.to_string()}")

    # Feature summary for FALL vs NORMAL
    print(f"\n--- acc_mag_mean by label ---")
    print(df.groupby("label")["acc_mag_mean"].describe().to_string())

    print(f"\n--- gyro_mag_mean by label ---")
    print(df.groupby("label")["gyro_mag_mean"].describe().to_string())

    # Save a compact inventory summary
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": df["label"].value_counts().to_dict(),
        "dataset_counts": df["dataset"].value_counts().to_dict(),
        "n_subjects": int(df["subject_id"].nunique()),
        "subjects": sorted(df["subject_id"].unique().tolist()),
        "label_by_dataset": ct.to_dict(),
    }
    out = Path("data/derived") / "fall_detection_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()