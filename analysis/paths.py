"""Single source of truth for filesystem locations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = ROOT / "models"
DATA = ROOT / "data"
DERIVED = DATA / "derived"

MCFD_ROOT = (
    DATA / "raw/_kagglehub/datasets/soumicksarker/multiple-cameras-fall-dataset/versions/2"
)
MCFD_VIDEOS = MCFD_ROOT / "dataset/dataset"
MCFD_LABELS = MCFD_ROOT / "data_tuple3.csv"

# Third dataset: CCTV incident (lying/standing), used to validate the posture
# signal on diverse scenes rather than temporal fall events.
CCTV_INCIDENT_ROOT = (
    DATA / "raw/_kagglehub/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection/versions/1"
)
CCTV_INCIDENT_LABELS = CCTV_INCIDENT_ROOT / "laying_dataset"
