"""16_ScoreAware_CalibratedEnsemble.

Create final portfolio submissions by combining:
  - existing model/blend candidates
  - 13_EOLHazardGate_Calibrator
  - 15_TrajectoryKNN_DTW_RUL

Outputs:
  results/16_scoreaware_debug.csv
  results/16_scoreaware_safe_submission.xlsx
  results/16_scoreaware_balanced_submission.xlsx
  results/16_scoreaware_aggressive_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import VAL_NAMES  # noqa: E402

RESULT_DIR = result_dir("16_ScoreAware_CalibratedEnsemble")
CANDIDATE_FILES = {
    "submission_v24_v17v22_debug.csv": RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv",
    "submission_v19_blend_debug.csv": RESULT_ROOT / "03_HIBlend_Baseline_EOLDirect" / "submission_v19_blend_debug.csv",
    "submission_v8_v17v25_debug.csv": RESULT_ROOT / "08_HIBlend_Baseline_Dynamics" / "submission_v8_v17v25_debug.csv",
    "submission_v9_v17v26_debug.csv": RESULT_ROOT / "09_HIBlend_Baseline_DomainAdv" / "submission_v9_v17v26_debug.csv",
}


def load_candidate(fname: str, method_name: str, pred_col: str) -> pd.DataFrame:
    p = CANDIDATE_FILES[fname]
    if not p.exists():
        return pd.DataFrame({"bearing": VAL_NAMES, method_name: np.nan})
    df = pd.read_csv(p)
    bearing_col = "Bearing" if "Bearing" in df.columns else "bearing"
    out = df[[bearing_col, pred_col]].copy()
    out.columns = ["bearing", method_name]
    return out


def make_submission(df: pd.DataFrame, col: str, fname: str) -> None:
    out = pd.DataFrame({
        "Bearing": df["bearing"],
        "RUL_pred_seconds": df[col].clip(lower=600.0),
    })
    out["RUL_pred_hours"] = out["RUL_pred_seconds"] / 3600.0
    out.to_excel(RESULT_DIR / fname, index=False)


def main() -> None:
    base = pd.DataFrame({"bearing": VAL_NAMES})
    candidates = [
        load_candidate("submission_v24_v17v22_debug.csv", "5_ChannelSymBlend_combined", "RUL_blend_combined_s"),
        load_candidate("submission_v24_v17v22_debug.csv", "5_ChannelSymBlend_full", "RUL_blend_full_s"),
        load_candidate("submission_v19_blend_debug.csv", "3_EOLDirectBlend_combined", "RUL_blend_combined_s"),
        load_candidate("submission_v8_v17v25_debug.csv", "8_DynamicsBlend_combined", "RUL_combined_s"),
        load_candidate("submission_v9_v17v26_debug.csv", "9_DomainAdvBlend_combined", "RUL_combined_s"),
        load_candidate("submission_v9_v17v26_debug.csv", "9_DomainAdvBlend_full", "RUL_full_s"),
    ]
    for c in candidates:
        base = base.merge(c, on="bearing", how="left")

    gate = pd.read_csv(RESULT_ROOT / "13_EOLHazardGate_Calibrator" / "13_eol_hazard_test.csv")
    gate_cols = [
        "bearing", "HI", "rms_multi", "energy_ratio", "nn1_bearing", "nn1_rul_s",
        "p_eol_2400", "p_eol_6000", "knn_rul_q35", "13_EOLHazardGate_safe_rul_s", "gate_reason",
    ]
    base = base.merge(gate[gate_cols], on="bearing", how="left")

    traj = pd.read_csv(RESULT_ROOT / "15_TrajectoryKNN_DTW_RUL" / "15_trajectory_knn_test.csv")
    traj_agg = traj.groupby("bearing", as_index=False).agg({
        "traj_rul_q20": "min",
        "traj_rul_q35": "min",
        "traj_rul_q50": "min",
        "p_eol_2400": "max",
        "p_eol_6000": "max",
        "nn1_rul_s": "min",
    }).rename(columns={
        "p_eol_2400": "traj_p_eol_2400",
        "p_eol_6000": "traj_p_eol_6000",
        "nn1_rul_s": "traj_nn1_min_rul_s",
    })
    base = base.merge(traj_agg, on="bearing", how="left")

    balanced = []
    safe = []
    aggressive = []
    reasons = []
    for _, row in base.iterrows():
        anchor = float(row["5_ChannelSymBlend_combined"])
        gate_pred = float(row["13_EOLHazardGate_safe_rul_s"])
        traj_q35 = float(row["traj_rul_q35"])
        traj_p2400 = float(row["traj_p_eol_2400"])
        traj_p6000 = float(row["traj_p_eol_6000"])
        gate_reason = str(row["gate_reason"])

        # Balanced: current stable best with hard EOL gate.
        b = min(anchor, gate_pred)

        # Safe: only additionally trust trajectory if independent trajectory risk is visible.
        s = b
        if gate_reason != "pass" or traj_p2400 >= 0.15 or traj_p6000 >= 0.25:
            s = min(s, max(600.0, traj_q35))

        # Aggressive: use domain-adv blend where no EOL risk, otherwise obey the same gate.
        a_anchor = float(row["9_DomainAdvBlend_combined"])
        a = a_anchor if gate_reason == "pass" and traj_p2400 < 0.15 and traj_p6000 < 0.25 else s

        balanced.append(max(600.0, b))
        safe.append(max(600.0, s))
        aggressive.append(max(600.0, a))
        reasons.append(f"gate={gate_reason}; traj_p2400={traj_p2400:.3f}; traj_p6000={traj_p6000:.3f}")

    base["16_balanced_rul_s"] = balanced
    base["16_safe_rul_s"] = safe
    base["16_aggressive_rul_s"] = aggressive
    base["16_reason"] = reasons
    base.to_csv(RESULT_DIR / "16_scoreaware_debug.csv", index=False)

    make_submission(base, "16_safe_rul_s", "16_scoreaware_safe_submission.xlsx")
    make_submission(base, "16_balanced_rul_s", "16_scoreaware_balanced_submission.xlsx")
    make_submission(base, "16_aggressive_rul_s", "16_scoreaware_aggressive_submission.xlsx")

    print("16_ScoreAware_CalibratedEnsemble")
    cols = [
        "bearing", "5_ChannelSymBlend_combined", "9_DomainAdvBlend_combined",
        "13_EOLHazardGate_safe_rul_s", "traj_rul_q35", "traj_p_eol_2400", "traj_p_eol_6000",
        "16_safe_rul_s", "16_balanced_rul_s", "16_aggressive_rul_s", "16_reason",
    ]
    print(base[cols].to_string(index=False))
    print(f"  Saved: {RESULT_DIR / '16_scoreaware_debug.csv'}")
    print(f"  Saved: {RESULT_DIR / '16_scoreaware_safe_submission.xlsx'}")
    print(f"  Saved: {RESULT_DIR / '16_scoreaware_balanced_submission.xlsx'}")
    print(f"  Saved: {RESULT_DIR / '16_scoreaware_aggressive_submission.xlsx'}")


if __name__ == "__main__":
    main()
