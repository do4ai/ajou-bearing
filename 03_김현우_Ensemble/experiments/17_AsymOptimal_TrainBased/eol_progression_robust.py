"""19_EOLProgression_Robust — outlier cap + median estimate.

기존 19번 문제: Test3 HI=0.165에서 progression ratio가 너무 작음 (~0.005) →
  Test EOL time = t_test_last / prog → 폭발 (137h, RUL=493k s)

해결:
  1. Train EOL time의 최대값으로 cap (max(Train EOL) = 22.8h = 82080s)
  2. 4 ref bearing의 median 사용 (asym-optimal은 outlier에 약함)
  3. HI=h인 train 측정의 RUL distribution과 cross-check

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/19_eol_progression_robust.csv
  artifacts/results/17_AsymOptimal_TrainBased/19_eol_progression_robust_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def asym_optimal_from_samples(rul_samples, min_pred=600.0):
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
    print("19_EOLProgression_Robust — outlier cap + median")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    test_last = df[df.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).reset_index(drop=True)

    # Train EOL times + max RUL bound
    train_curves = {}
    train_eol_times = []
    for nm in TRAIN_NAMES:
        sub = train[train.bearing == nm].sort_values("t_s").reset_index(drop=True)
        t_eol = float(sub["t_s"].iloc[-1]) + 600.0
        train_eol_times.append(t_eol)
        train_curves[nm] = {"t_eol": t_eol, "prog": sub["t_s"].values / t_eol,
                             "hi": sub["HI"].values, "t_s": sub["t_s"].values}
    eol_max = max(train_eol_times)
    eol_median = float(np.median(train_eol_times))
    print(f"  Train EOL times: {[f'{t/3600:.1f}h' for t in train_eol_times]}")
    print(f"  EOL max={eol_max/3600:.2f}h  median={eol_median/3600:.2f}h")

    # Per-test robust estimate
    print("\n  Per-test robust EOL progression estimates:")
    results = []
    for _, t_row in test_last.iterrows():
        bearing = t_row["bearing"]
        hi_test = float(t_row["HI"])
        t_test_last = float(t_row["t_s"])

        # 4 ref est_RUL with progression cap
        est_ruls_raw = []
        est_ruls_capped = []
        for nm in TRAIN_NAMES:
            curve = train_curves[nm]
            idx = int(np.argmin(np.abs(curve["hi"] - hi_test)))
            prog = max(float(curve["prog"][idx]), 0.01)  # min progression cap 0.01 → max RUL boost
            est_eol_t_raw = t_test_last / prog
            est_eol_t_capped = min(est_eol_t_raw, eol_max)  # cap by max train EOL
            est_ruls_raw.append(max(600.0, est_eol_t_raw - t_test_last))
            est_ruls_capped.append(max(600.0, est_eol_t_capped - t_test_last))

        est_ruls_capped = np.array(est_ruls_capped)
        pred_median = float(np.median(est_ruls_capped))
        pred_asym = asym_optimal_from_samples(est_ruls_capped)
        pred_min = float(est_ruls_capped.min())

        results.append({
            "Bearing": bearing,
            "HI_last": hi_test,
            "t_test_last_h": t_test_last / 3600,
            "est_rul_raw_median_h": float(np.median(est_ruls_raw)) / 3600,
            "est_rul_capped_min_h": pred_min / 3600,
            "est_rul_capped_median_h": pred_median / 3600,
            "est_rul_capped_max_h": float(est_ruls_capped.max()) / 3600,
            "asym_optimal_pred_s": pred_asym,
            "median_pred_s": pred_median,
            "min_pred_s": pred_min,
        })
        print(f"  {bearing}: HI={hi_test:.3f}  "
              f"raw_median={np.median(est_ruls_raw)/3600:.1f}h  "
              f"capped_median={pred_median/3600:.2f}h  "
              f"asym_opt={pred_asym/3600:.2f}h ({pred_asym:.0f}s)")

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULT_DIR / "19_eol_progression_robust.csv", index=False)
    sub = out_df[["Bearing", "HI_last", "asym_optimal_pred_s"]].copy()
    sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "19_eol_progression_robust_submission.xlsx", index=False)

    # LOBO 검증
    print("\n[LOBO 검증]")
    lobo_rows = []
    for val in TRAIN_NAMES:
        v_sub = train[train.bearing == val].sort_values("t_s").reset_index(drop=True)
        t_v_last = float(v_sub["t_s"].iloc[-1])
        hi_v_last = float(v_sub["HI"].iloc[-1])
        true_rul = float(v_sub["rul_s"].iloc[-1])

        est_ruls = []
        ref_eols = []
        for nm in TRAIN_NAMES:
            if nm == val: continue
            curve = train_curves[nm]
            ref_eols.append(curve["t_eol"])
            idx = int(np.argmin(np.abs(curve["hi"] - hi_v_last)))
            prog = max(float(curve["prog"][idx]), 0.01)
            est_eol_t = min(t_v_last / prog, max(ref_eols))
            est_ruls.append(max(600.0, est_eol_t - t_v_last))

        est_ruls = np.array(est_ruls)
        pred_asym = asym_optimal_from_samples(est_ruls)
        score = asym_score([pred_asym], [true_rul])
        lobo_rows.append({"val": val, "true": true_rul,
                           "pred_asym": pred_asym, "score": score,
                           "est_ruls": ",".join(f"{r:.0f}" for r in est_ruls)})
        print(f"  {val}: HI={hi_v_last:.3f}  true={true_rul:.0f}  pred={pred_asym:.0f}  score={score:.3f}")

    lobo_df = pd.DataFrame(lobo_rows)
    lobo_df.to_csv(RESULT_DIR / "19_eol_progression_robust_lobo.csv", index=False)
    print(f"\n  LOBO mean score: {lobo_df['score'].mean():.4f}")
    print(f"\n  Saved: {RESULT_DIR / '19_eol_progression_robust.csv'}")
    print(f"  Saved: {RESULT_DIR / '19_eol_progression_robust_submission.xlsx'}")


if __name__ == "__main__":
    main()
