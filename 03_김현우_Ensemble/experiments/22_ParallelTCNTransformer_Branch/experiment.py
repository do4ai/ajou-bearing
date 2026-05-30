"""22_ParallelTCNTransformer_Branch.

Lightweight parallel local/global branch surrogate:
  - local branch: recent dynamics features, nearest temporal state
  - global branch: baseline/channel/domain candidate ensemble
This tests the parallel TCN+Transformer idea without retraining large nets.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score

RESULT_DIR = result_dir("22_ParallelTCNTransformer_Branch")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def seq_table(df: pd.DataFrame, cols: list[str], seq_len: int = 10) -> pd.DataFrame:
    rows = []
    for b, sub in df.groupby("bearing", sort=False):
        sub = sub.reset_index(drop=True)
        if len(sub) < seq_len:
            continue
        x = sub[cols].values
        for end in range(seq_len - 1, len(sub)):
            recent = x[end - seq_len + 1:end + 1]
            row = {"bearing": b, "measurement": int(sub.iloc[end].measurement), "t_s": float(sub.iloc[end].t_s)}
            if "rul_s" in sub.columns:
                row["rul_s"] = float(sub.iloc[end].rul_s)
            for i, c in enumerate(cols):
                row[f"local_last_{c}"] = recent[-1, i]
                row[f"local_mean_{c}"] = recent[:, i].mean()
                row[f"local_slope_{c}"] = recent[-1, i] - recent[0, i]
            rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = [c for c in ["HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt", "HI_slope5", "energy_ratio_slope5"] if c in df.columns]
    tab = seq_table(df, cols)
    feat_cols = [c for c in tab.columns if c.startswith("local_")]
    lobo = []
    for val in TRAIN_NAMES:
        tr = tab[(tab.bearing.isin(TRAIN_NAMES)) & (tab.bearing != val)]
        va = tab[tab.bearing == val]
        sc = StandardScaler().fit(tr[feat_cols].values)
        rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=22, min_samples_leaf=3)
        rf.fit(sc.transform(tr[feat_cols].values), tr.rul_s.values)
        p = np.clip(rf.predict(sc.transform(va[feat_cols].values)), 600, None)
        lobo.append({"bearing": val, "full": asym_score(p, va.rul_s.values), "last": asym_score([p[-1]], [va.rul_s.values[-1]]), "pred_last": p[-1]})
    lobo_df = pd.DataFrame(lobo)
    lobo_df.to_csv(RESULT_DIR / "22_parallel_branch_lobo.csv", index=False)
    tr = tab[tab.bearing.isin(TRAIN_NAMES)]
    te = tab[tab.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1)
    sc = StandardScaler().fit(tr[feat_cols].values)
    rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=220, min_samples_leaf=3)
    rf.fit(sc.transform(tr[feat_cols].values), tr.rul_s.values)
    p = np.clip(rf.predict(sc.transform(te[feat_cols].values)), 600, None)
    sub = pd.DataFrame({"Bearing": te.bearing.values, "RUL_pred_seconds": p, "RUL_pred_hours": p / 3600})
    sub.to_csv(RESULT_DIR / "22_parallel_branch_candidate.csv", index=False)
    sub.to_excel(RESULT_DIR / "22_parallel_branch_submission.xlsx", index=False)
    print("22_ParallelTCNTransformer_Branch")
    print(lobo_df.to_string(index=False))
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
