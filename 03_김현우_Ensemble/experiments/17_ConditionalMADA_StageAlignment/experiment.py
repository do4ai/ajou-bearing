"""17_ConditionalMADA_StageAlignment.

Lightweight MADA diagnostic: assign Train stages from true RUL, assign Test
pseudo stages from HI/energy risk, then compute source-to-target discrepancy per
stage. This does not train a new adversarial net; it tells which Train bearing
should dominate target adaptation.
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

RESULT_DIR = result_dir("17_ConditionalMADA_StageAlignment")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def stage_from_rul(rul: float, max_rul: float) -> str:
    ratio = rul / max(max_rul, 1.0)
    if ratio <= 0.08:
        return "eol"
    if ratio <= 0.33:
        return "late"
    if ratio <= 0.66:
        return "mid"
    return "early"


def pseudo_stage(row: pd.Series) -> str:
    if row["HI"] >= 0.90 or (row["energy_ratio"] >= 20 and row["rms_multi"] >= 0.5):
        return "eol"
    if row["HI"] >= 0.70 or row["energy_ratio"] >= 10:
        return "late"
    if row["HI"] >= 0.35:
        return "mid"
    return "early"


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("inf")
    xy = np.vstack([x, y])
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / (med + 1e-12)
    kxx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)).mean()
    kyy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)).mean()
    kxy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)).mean()
    return float(kxx + kyy - 2 * kxy)


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = [c for c in df.columns if c not in {"bearing", "measurement", "t_s", "rul_s"} and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c == "HI" or c.startswith(("rms", "energy", "chsym")) or any(k in c for k in ["_d", "_slope", "_roll_std", "_acc"])]
    sc = StandardScaler().fit(df[df.bearing.isin(TRAIN_NAMES)][cols].values)

    train = df[df.bearing.isin(TRAIN_NAMES)].copy()
    test = df[df.bearing.isin(VAL_NAMES)].copy()
    train["stage"] = "early"
    for b in TRAIN_NAMES:
        mask = train.bearing == b
        max_rul = float(train.loc[mask, "rul_s"].max())
        train.loc[mask, "stage"] = [stage_from_rul(r, max_rul) for r in train.loc[mask, "rul_s"]]
    test_last = test.groupby("bearing", sort=False).tail(1).copy()
    test_last["pseudo_stage"] = [pseudo_stage(r) for _, r in test_last.iterrows()]

    rows = []
    x_train_all = pd.DataFrame(sc.transform(train[cols].values), columns=cols)
    train = train.reset_index(drop=True)
    for _, tr in test_last.iterrows():
        stage = tr["pseudo_stage"]
        xt = sc.transform(tr[cols].values.reshape(1, -1))
        for src in TRAIN_NAMES:
            pool = train[(train.bearing == src) & (train.stage == stage)]
            if len(pool) < 3:
                pool = train[train.bearing == src]
            xs = x_train_all.loc[pool.index].values
            dist = mmd_rbf(xs, xt)
            rows.append({"target": tr["bearing"], "pseudo_stage": stage, "source": src, "mmd": dist, "n_source": len(pool)})
    out = pd.DataFrame(rows)
    out["weight"] = out.groupby("target")["mmd"].transform(lambda s: (1 / (s + 1e-6)) / (1 / (s + 1e-6)).sum())
    out.to_csv(RESULT_DIR / "17_conditional_mada_stage_weights.csv", index=False)
    summary = out.sort_values(["target", "weight"], ascending=[True, False]).groupby("target").head(2)
    summary.to_csv(RESULT_DIR / "17_conditional_mada_top_sources.csv", index=False)
    print("17_ConditionalMADA_StageAlignment")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
