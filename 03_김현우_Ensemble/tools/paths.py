"""Shared paths for 03_김현우_Ensemble after folder cleanup."""
from pathlib import Path
import sys

ENSEMBLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ENSEMBLE_DIR.parents[0]
ARTIFACT_DIR = ENSEMBLE_DIR / "artifacts"
MODEL_ROOT = ARTIFACT_DIR / "models"
RESULT_ROOT = ARTIFACT_DIR / "results"
LOG_ROOT = ARTIFACT_DIR / "logs"


def add_repo_to_path() -> None:
    for p in [str(REPO_ROOT), str(ENSEMBLE_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def result_dir(name: str) -> Path:
    p = RESULT_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(name: str) -> Path:
    return MODEL_ROOT / name
