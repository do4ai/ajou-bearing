"""16_ScoreAware_CalibratedEnsemble — train-based redesign.

핵심 원칙 (사용자 지적 반영):
  모든 RUL 출력은 Train data로 학습된 모델의 회귀 결과여야 한다.
  임의 숫자(2400/3600/6000 같은 hard-coded) 박기 절대 금지.
  600s 물리 하한 클립만 허용 (측정 간격 자체가 하한).

구조:
  - Anchor: 5_HIBlend_Baseline_ChannelSym (current stable best, LOBO combined 0.712)
  - EOL specialist: 28_EOLRegressor_Specialist (Train rul_s≤15000 학습)
  - Trajectory KNN: 15_TrajectoryKNN_DTW_RUL (weighted q35 from Train RUL distribution)
  - Gate signal: 13_EOLHazardGate_Calibrator (P_eol_2400, P_eol_6000) — 값 X, regime 선택만

Regime 결정:
  P_eol 강도에 따라 3-way weighted ensemble 가중치 변경. 임의 clamp 없음.
  - strong (hi≥0.90 or p2400≥0.30): EOL 40%, KNN 30%, anchor 30%
  - moderate (p2400≥0.20 or energy≥15+p6000≥0.20): EOL 30%, KNN 40%, anchor 30%
  - weak (p3600≥0.30 or energy≥20): EOL 20%, KNN 30%, anchor 50%
  - pass: anchor 100%

Outputs:
  results/16_scoreaware_debug.csv
  results/16_scoreaware_balanced_submission.xlsx  (default, train-based 3-way)
  results/16_scoreaware_safe_submission.xlsx      (보수 측 quantile weight)
  results/16_scoreaware_aggressive_submission.xlsx (anchor heavy)
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


def load_train_based_predictions() -> pd.DataFrame:
    """Train-based 4가지 예측값을 베어링별로 모은다.

    - anchor: 5_HIBlend_Baseline_ChannelSym
    - eol_reg: 28_EOLRegressor_Specialist (conservative)
    - knn_q35: 15_TrajectoryKNN_DTW_RUL (window 10/20 중 최소 q35 = 보수)
    - gate features: 13_EOLHazardGate (HI, energy_ratio, p_eol_*)
    """
    # 1) anchor (5_HIBlend) — 현재 best stable
    anchor_path = RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv"
    anchor = pd.read_csv(anchor_path)[["Bearing", "RUL_blend_combined_s", "RUL_blend_full_s"]]
    anchor = anchor.rename(columns={
        "Bearing": "bearing",
        "RUL_blend_combined_s": "anchor_combined",
        "RUL_blend_full_s": "anchor_full",
    })

    # 2) EOL specialist (28) — train-based EOL regression
    eol_path = RESULT_ROOT / "28_EOLRegressor_Specialist" / "28_eol_regressor_test.csv"
    eol = pd.read_csv(eol_path)[["Bearing", "EOL_conservative_s", "EOL_median_s",
                                  "EOL_gbm_q4_s", "EOL_rf_s"]]
    eol = eol.rename(columns={"Bearing": "bearing"})

    # 3) trajectory KNN (15) — train RUL distribution
    knn_path = RESULT_ROOT / "15_TrajectoryKNN_DTW_RUL" / "15_trajectory_knn_test.csv"
    knn_raw = pd.read_csv(knn_path)
    knn = knn_raw.groupby("bearing", as_index=False).agg({
        "traj_rul_q20": "min",
        "traj_rul_q35": "min",
        "traj_rul_q50": "min",
        "p_eol_2400": "max",
        "p_eol_6000": "max",
        "nn1_rul_s": "min",
    }).rename(columns={
        "traj_rul_q20": "knn_q20",
        "traj_rul_q35": "knn_q35",
        "traj_rul_q50": "knn_q50",
        "p_eol_2400": "knn_p_eol_2400",
        "p_eol_6000": "knn_p_eol_6000",
        "nn1_rul_s": "knn_nn1_min_rul",
    })

    # 4) gate (13) — P_eol classifier features
    gate_path = RESULT_ROOT / "13_EOLHazardGate_Calibrator" / "13_eol_hazard_test.csv"
    gate = pd.read_csv(gate_path)[["bearing", "HI", "rms_multi", "energy_ratio",
                                    "p_eol_2400", "p_eol_3600", "p_eol_6000",
                                    "knn_rul_q35"]]
    gate = gate.rename(columns={
        "HI": "gate_HI",
        "rms_multi": "gate_rms_multi",
        "energy_ratio": "gate_energy_ratio",
        "p_eol_2400": "gate_p_eol_2400",
        "p_eol_3600": "gate_p_eol_3600",
        "p_eol_6000": "gate_p_eol_6000",
        "knn_rul_q35": "gate_knn_q35",
    })

    df = anchor.merge(eol, on="bearing", how="outer") \
               .merge(knn, on="bearing", how="outer") \
               .merge(gate, on="bearing", how="outer")
    return df


def regime_decision(row: pd.Series) -> tuple[float, float, float, str]:
    """Train-based 3-way ensemble. 임의 숫자 박기 금지.

    Returns (balanced, safe, aggressive, regime_label).
    """
    anchor = float(row["anchor_combined"])
    eol = float(row["EOL_conservative_s"])
    eol_q4 = float(row["EOL_gbm_q4_s"])
    knn_q35 = float(row["knn_q35"])
    knn_q20 = float(row["knn_q20"])

    hi = float(row["gate_HI"])
    energy = float(row["gate_energy_ratio"])
    p2400 = float(row["gate_p_eol_2400"])
    p3600 = float(row["gate_p_eol_3600"])
    p6000 = float(row["gate_p_eol_6000"])

    # Regime 분류
    if hi >= 0.90 or p2400 >= 0.30:
        regime = "strong_eol"
    elif p2400 >= 0.20 or (energy >= 15.0 and p6000 >= 0.20):
        regime = "moderate_eol"
    elif p3600 >= 0.30 or energy >= 20.0 or p6000 >= 0.25:
        regime = "weak_eol"
    else:
        regime = "pass"

    # Train-based weighted ensembles
    if regime == "strong_eol":
        balanced = 0.40 * eol + 0.30 * knn_q35 + 0.30 * anchor
        safe = 0.35 * eol_q4 + 0.45 * knn_q20 + 0.20 * anchor
        aggressive = 0.50 * anchor + 0.30 * eol + 0.20 * knn_q35
    elif regime == "moderate_eol":
        balanced = 0.30 * eol + 0.40 * knn_q35 + 0.30 * anchor
        safe = 0.30 * eol_q4 + 0.50 * knn_q20 + 0.20 * anchor
        aggressive = 0.60 * anchor + 0.25 * eol + 0.15 * knn_q35
    elif regime == "weak_eol":
        balanced = 0.20 * eol + 0.30 * knn_q35 + 0.50 * anchor
        safe = 0.25 * eol_q4 + 0.35 * knn_q20 + 0.40 * anchor
        aggressive = 0.80 * anchor + 0.15 * eol + 0.05 * knn_q35
    else:  # pass
        balanced = anchor
        safe = 0.85 * anchor + 0.10 * eol_q4 + 0.05 * knn_q20
        aggressive = anchor

    # 600s 물리 하한만 (측정 간격 = 물리적 하한, 임의 안 됨)
    balanced = max(600.0, balanced)
    safe = max(600.0, safe)
    aggressive = max(600.0, aggressive)

    return balanced, safe, aggressive, regime


def make_submission_xlsx(df: pd.DataFrame, col: str, fname: str) -> Path:
    out = pd.DataFrame({
        "Bearing": df["bearing"],
        "HI_last": df["gate_HI"],
        "RUL_pred_seconds": df[col].astype(float),
    })
    out["RUL_pred_hours"] = out["RUL_pred_seconds"] / 3600.0
    fp = RESULT_DIR / fname
    out.to_excel(fp, index=False)
    return fp


def main() -> None:
    print("=" * 70)
    print("16_ScoreAware_CalibratedEnsemble (Train-based redesign)")
    print("=" * 70)

    df = load_train_based_predictions()

    bal, saf, agr, reg = [], [], [], []
    for _, row in df.iterrows():
        b, s, a, r = regime_decision(row)
        bal.append(b); saf.append(s); agr.append(a); reg.append(r)

    df["regime"] = reg
    df["balanced_rul_s"] = bal
    df["safe_rul_s"] = saf
    df["aggressive_rul_s"] = agr

    # 정렬 (원래 Test1~6 순서 유지)
    df["_order"] = df["bearing"].map({n: i for i, n in enumerate(VAL_NAMES)})
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    show_cols = ["bearing", "gate_HI", "anchor_combined", "EOL_conservative_s",
                 "knn_q35", "gate_p_eol_2400", "gate_p_eol_6000",
                 "regime", "balanced_rul_s", "safe_rul_s", "aggressive_rul_s"]
    print(df[show_cols].to_string(index=False))

    df.to_csv(RESULT_DIR / "16_scoreaware_debug.csv", index=False)

    bal_fp = make_submission_xlsx(df, "balanced_rul_s", "16_scoreaware_balanced_submission.xlsx")
    saf_fp = make_submission_xlsx(df, "safe_rul_s", "16_scoreaware_safe_submission.xlsx")
    agr_fp = make_submission_xlsx(df, "aggressive_rul_s", "16_scoreaware_aggressive_submission.xlsx")

    print()
    print(f"  Saved: {RESULT_DIR / '16_scoreaware_debug.csv'}")
    print(f"  Saved: {bal_fp}")
    print(f"  Saved: {saf_fp}")
    print(f"  Saved: {agr_fp}")
    print("\n  Note: 모든 출력은 train-based (5_HIBlend / 28_EOLRegressor / 15_KNN)의 weighted combination.")
    print("        임의 숫자 박기 없음. 600s 물리 하한 클립만 적용.")


if __name__ == "__main__":
    main()
