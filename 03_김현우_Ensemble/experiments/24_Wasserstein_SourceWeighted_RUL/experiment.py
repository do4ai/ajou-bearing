"""24_Wasserstein_SourceWeighted_RUL.

Wasserstein-distance source weighting for Test RUL calibration.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES

RESULT_DIR = result_dir("24_Wasserstein_SourceWeighted_RUL")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = [c for c in ["HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt", "HI_slope5"] if c in df.columns]
    train = df[df.bearing.isin(TRAIN_NAMES)].copy()
    sc = StandardScaler().fit(train[cols].values)
    xtr = sc.transform(train[cols].values)
    rows, subs = [], []
    for target in VAL_NAMES:
        tsub = df[df.bearing == target].tail(15)
        xt = sc.transform(tsub[cols].values)
        src_d = []
        for src in TRAIN_NAMES:
            ssub = df[df.bearing == src]
            xs = sc.transform(ssub[cols].values)
            d = np.mean([wasserstein_distance(xs[:, j], xt[:, j]) for j in range(len(cols))])
            src_d.append((src, d))
            rows.append({"target": target, "source": src, "wasserstein": d})
        weights = {s: (1 / (d + 1e-6)) for s, d in src_d}
        z = sum(weights.values())
        weights = {s: w / z for s, w in weights.items()}
        q = df[df.bearing == target].tail(1)[cols]
        xq = sc.transform(q.values)
        pred_parts = []
        for src, sw in weights.items():
            pool = train[train.bearing == src]
            xp = sc.transform(pool[cols].values)
            dist = pairwise_distances(xq, xp)[0]
            idx = np.argsort(dist)[:10]
            pred_parts.append(sw * np.quantile(pool.iloc[idx].rul_s.values, 0.35))
        pred = max(600.0, float(sum(pred_parts)))
        subs.append({"Bearing": target, "RUL_pred_seconds": pred, "RUL_pred_hours": pred / 3600})
    weights_df = pd.DataFrame(rows)
    weights_df["weight"] = weights_df.groupby("target")["wasserstein"].transform(lambda s: (1/(s+1e-6))/(1/(s+1e-6)).sum())
    weights_df.to_csv(RESULT_DIR / "24_wasserstein_source_weights.csv", index=False)
    sub = pd.DataFrame(subs)
    sub.to_csv(RESULT_DIR / "24_wasserstein_candidate.csv", index=False)
    sub.to_excel(RESULT_DIR / "24_wasserstein_submission.xlsx", index=False)
    print("24_Wasserstein_SourceWeighted_RUL")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
