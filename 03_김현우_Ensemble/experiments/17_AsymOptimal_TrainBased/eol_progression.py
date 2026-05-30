"""19_EOLProgression_TrainBased — HI 곡선 fitting으로 Test EOL 진행도 추정.

가설: HI는 시간에 따라 monotonic 증가 (DTC-VAE 학습 결과).
  Train: HI(t)는 진행도 ratio = t/T_eol로 표현 가능.
  Test: HI_last → progression ratio → Test EOL time 역산 → Test RUL.

알고리즘:
  1. Train1~4 각 베어링에서 (HI, progression_ratio) pair 수집.
     progression_ratio = (t_measurement / T_eol_train).
  2. ProgressionFit: HI = f(ratio). 다항식 또는 monotonic spline.
  3. Test마다 HI_last → f^-1(HI) = estimated_progression_ratio.
  4. Test estimated EOL time = t_test_last / progression_ratio.
  5. Test RUL = estimated_EOL_time - t_test_last.

Train의 4개 베어링 변이 -> bootstrap으로 distribution 추정.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/19_eol_progression.csv
  artifacts/results/17_AsymOptimal_TrainBased/19_eol_progression_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def asym_optimal_from_samples(rul_samples: np.ndarray, min_pred: float = 600.0) -> float:
    """RUL 분포에서 비대칭 최적 단일 prediction."""
    max_pred = float(rul_samples.max() * 1.2 + 1000)
    def neg_obj(p):
        er = 100.0 * (rul_samples - p) / (rul_samples + 1e-12)
        ln_half = np.log(0.5)
        arg_late = np.clip(-ln_half * er / 20.0, -50, 0)
        arg_early = np.clip(ln_half * er / 50.0, -50, 0)
        a = np.where(er <= 0, np.exp(arg_late), np.exp(arg_early))
        return -float(np.mean(a))
    res = minimize_scalar(neg_obj, bounds=(min_pred, max_pred), method="bounded")
    return float(res.x)


def main() -> None:
    print("=" * 70)
    print("19_EOLProgression — HI 곡선 fitting + Test EOL 진행도 추정")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    test_last = df[df.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).reset_index(drop=True)

    # Step 1: Train 각 베어링의 (progression_ratio, HI) 수집
    train_curves = {}
    for nm in TRAIN_NAMES:
        sub = train[train.bearing == nm].sort_values("t_s").reset_index(drop=True)
        t_eol = float(sub["t_s"].iloc[-1]) + 600.0  # 마지막 측정 + 600s = EOL 가정
        prog_ratios = sub["t_s"].values / t_eol
        hi_vals = sub["HI"].values
        train_curves[nm] = {
            "t_eol": t_eol, "n": len(sub),
            "prog": prog_ratios, "hi": hi_vals,
        }
        print(f"  {nm}: T_eol={t_eol/3600:.2f}h  n={len(sub)}  HI range [{hi_vals.min():.3f}, {hi_vals.max():.3f}]")

    # Step 2: Test에 대해 4개 train baseline 각각 적용 → 4개 estimated EOL 후보
    print("\n  Per-test EOL progression estimates:")
    results = []
    for _, t_row in test_last.iterrows():
        bearing = t_row["bearing"]
        hi_test = float(t_row["HI"])
        t_test_last = float(t_row["t_s"])

        est_eol_candidates = []
        for nm, curve in train_curves.items():
            # 진행 ratio → HI는 거의 monotonic이므로 invert 가능
            # numerical approach: HI=hi_test에 가장 가까운 train 측정 → 그 progression ratio
            idx = int(np.argmin(np.abs(curve["hi"] - hi_test)))
            prog = float(curve["prog"][idx])
            if prog < 1e-3:
                prog = 1e-3
            # Test estimated EOL time
            est_eol_t = t_test_last / prog
            est_eol_candidates.append({
                "ref_bearing": nm, "matched_hi": float(curve["hi"][idx]),
                "matched_prog": prog, "matched_train_t": float(curve["t_s"][idx]) if "t_s" in curve else 0.0,
                "est_eol_time_s": est_eol_t,
                "est_rul_s": max(600.0, est_eol_t - t_test_last),
            })

        est_ruls = np.array([c["est_rul_s"] for c in est_eol_candidates])
        # 비대칭 최적 prediction from these 4 estimates
        pred_asym = asym_optimal_from_samples(est_ruls)

        results.append({
            "Bearing": bearing,
            "HI_last": hi_test,
            "t_test_last_s": t_test_last,
            "ref_Train1_est_rul": est_eol_candidates[0]["est_rul_s"],
            "ref_Train2_est_rul": est_eol_candidates[1]["est_rul_s"],
            "ref_Train3_est_rul": est_eol_candidates[2]["est_rul_s"],
            "ref_Train4_est_rul": est_eol_candidates[3]["est_rul_s"],
            "est_rul_median_s": float(np.median(est_ruls)),
            "est_rul_min_s": float(est_ruls.min()),
            "est_rul_max_s": float(est_ruls.max()),
            "est_rul_q25_s": float(np.percentile(est_ruls, 25)),
            "est_rul_q40_s": float(np.percentile(est_ruls, 40)),
            "asym_optimal_pred_s": pred_asym,
        })
        print(f"  {bearing}: HI={hi_test:.3f}  t_test={t_test_last/3600:.2f}h  "
              f"est_RUL [Train1={est_eol_candidates[0]['est_rul_s']:.0f}, "
              f"Train2={est_eol_candidates[1]['est_rul_s']:.0f}, "
              f"Train3={est_eol_candidates[2]['est_rul_s']:.0f}, "
              f"Train4={est_eol_candidates[3]['est_rul_s']:.0f}]  "
              f"asym_opt={pred_asym:.0f}s")

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULT_DIR / "19_eol_progression.csv", index=False)

    sub = out_df[["Bearing", "HI_last", "asym_optimal_pred_s"]].copy()
    sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "19_eol_progression_submission.xlsx", index=False)
    print(f"\n  Saved: {RESULT_DIR / '19_eol_progression.csv'}")
    print(f"  Saved: {RESULT_DIR / '19_eol_progression_submission.xlsx'}")

    # LOBO 검증: train의 한 베어링을 빼고 나머지 3개로 fit → 해당 베어링의 마지막 측정 RUL 예측
    print("\n[LOBO 검증]")
    lobo_rows = []
    for val in TRAIN_NAMES:
        v_sub = train[train.bearing == val].sort_values("t_s").reset_index(drop=True)
        t_v_last = float(v_sub["t_s"].iloc[-1])
        hi_v_last = float(v_sub["HI"].iloc[-1])
        true_rul = float(v_sub["rul_s"].iloc[-1])

        est_ruls = []
        for nm in TRAIN_NAMES:
            if nm == val: continue
            curve = train_curves[nm]
            idx = int(np.argmin(np.abs(curve["hi"] - hi_v_last)))
            prog = float(curve["prog"][idx])
            if prog < 1e-3: prog = 1e-3
            est_eol_t = t_v_last / prog
            est_ruls.append(max(600.0, est_eol_t - t_v_last))

        pred = asym_optimal_from_samples(np.array(est_ruls))
        score = asym_score([pred], [true_rul])
        lobo_rows.append({"val": val, "true": true_rul, "pred": pred, "score": score,
                           "est_ruls": ",".join(f"{r:.0f}" for r in est_ruls)})
        print(f"  {val}: HI={hi_v_last:.3f}  true={true_rul:.0f}  pred={pred:.0f}  score={score:.3f}")

    lobo_df = pd.DataFrame(lobo_rows)
    lobo_df.to_csv(RESULT_DIR / "19_eol_progression_lobo.csv", index=False)
    print(f"\n  LOBO mean score: {lobo_df['score'].mean():.4f}")


if __name__ == "__main__":
    main()
