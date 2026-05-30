"""25_StageAwareTransformer_DA.

Stage-aware transformer-lite diagnostic. Uses attention over recent feature
sequence and stage-specific quantile calibration; avoids heavy full retraining.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES

RESULT_DIR = result_dir("25_StageAwareTransformer_DA")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def pseudo_stage(row):
    hi = float(row.get("HI", row.get("last_HI", 0.0)))
    energy = float(row.get("energy_ratio", row.get("last_energy_ratio", 0.0)))
    if hi >= 0.9 or energy >= 20:
        return 3
    if hi >= 0.7 or energy >= 10:
        return 2
    if hi >= 0.35:
        return 1
    return 0


def stage_from_rul(r, maxr):
    q = r / max(maxr, 1)
    return 3 if q <= 0.08 else 2 if q <= 0.33 else 1 if q <= 0.66 else 0


def seq_rows(df, cols, seq_len=10):
    rows = []
    for b, sub in df.groupby("bearing", sort=False):
        sub = sub.reset_index(drop=True)
        for end in range(seq_len - 1, len(sub)):
            recent = sub.iloc[end-seq_len+1:end+1]
            score = recent["HI"].values + 0.2 * np.log1p(recent["energy_ratio"].values)
            w = np.exp(score - score.max()); w = w / w.sum()
            row = {"bearing": b, "measurement": int(sub.iloc[end].measurement)}
            if "rul_s" in sub.columns:
                row["rul_s"] = float(sub.iloc[end].rul_s)
            for c in cols:
                v = recent[c].values
                row[f"attn_{c}"] = float(np.sum(w * v))
                row[f"last_{c}"] = float(v[-1])
            rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = [c for c in ["HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt", "HI_slope5"] if c in df.columns]
    tab = seq_rows(df, cols)
    train = tab[tab.bearing.isin(TRAIN_NAMES)].copy()
    train["stage"] = 0
    for b in TRAIN_NAMES:
        m = train.bearing == b
        maxr = train.loc[m, "rul_s"].max()
        train.loc[m, "stage"] = [stage_from_rul(r, maxr) for r in train.loc[m, "rul_s"]]
    feat = [c for c in tab.columns if c.startswith(("attn_", "last_"))]
    test = tab[tab.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).copy()
    test["stage"] = [pseudo_stage(r) for _, r in test.iterrows()]
    preds = []
    for _, r in test.iterrows():
        pool = train[train.stage == r.stage]
        if len(pool) < 20:
            pool = train
        sc = StandardScaler().fit(pool[feat].values)
        model = GradientBoostingRegressor(loss="quantile", alpha=0.35, random_state=25, max_depth=2, n_estimators=120)
        model.fit(sc.transform(pool[feat].values), pool.rul_s.values)
        pred = max(600.0, float(model.predict(sc.transform(r[feat].values.reshape(1, -1)))[0]))
        preds.append({"Bearing": r.bearing, "stage": int(r.stage), "RUL_pred_seconds": pred, "RUL_pred_hours": pred/3600})
    sub = pd.DataFrame(preds)
    sub.to_csv(RESULT_DIR / "25_stageaware_transformer_da_candidate.csv", index=False)
    sub.to_excel(RESULT_DIR / "25_stageaware_transformer_da_submission.xlsx", index=False)
    print("25_StageAwareTransformer_DA")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
