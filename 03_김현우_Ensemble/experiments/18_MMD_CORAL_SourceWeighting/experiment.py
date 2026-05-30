"""18_MMD_CORAL_SourceWeighting.

Non-adversarial multi-source domain adaptation diagnostic using MMD + CORAL.
Outputs per-Test source weights and a source-risk table.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES

RESULT_DIR = result_dir("18_MMD_CORAL_SourceWeighting")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def coral(x: np.ndarray, y: np.ndarray) -> float:
    cx = np.cov(x, rowvar=False) if len(x) > 1 else np.eye(x.shape[1])
    cy = np.cov(y, rowvar=False) if len(y) > 1 else np.eye(y.shape[1])
    return float(np.linalg.norm(cx - cy, ord="fro") / (4 * x.shape[1] ** 2))


def mean_mmd(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)))


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = [c for c in ["HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt", "HI_slope5", "energy_ratio_slope5"] if c in df.columns]
    sc = StandardScaler().fit(df[df.bearing.isin(TRAIN_NAMES)][cols].values)
    rows = []
    for target in VAL_NAMES:
        tsub = df[df.bearing == target].tail(15)
        yt = sc.transform(tsub[cols].values)
        for src in TRAIN_NAMES:
            ssub = df[df.bearing == src]
            xs = sc.transform(ssub[cols].values)
            d_mmd = mean_mmd(xs, yt)
            d_coral = coral(xs, yt)
            d = d_mmd + d_coral
            rows.append({"target": target, "source": src, "mean_mmd": d_mmd, "coral": d_coral, "distance": d})
    out = pd.DataFrame(rows)
    out["weight"] = out.groupby("target")["distance"].transform(lambda s: (1 / (s + 1e-6)) / (1 / (s + 1e-6)).sum())
    out.to_csv(RESULT_DIR / "18_mmd_coral_source_weights.csv", index=False)
    print("18_MMD_CORAL_SourceWeighting")
    print(out.sort_values(["target", "weight"], ascending=[True, False]).groupby("target").head(2).to_string(index=False))


if __name__ == "__main__":
    main()
