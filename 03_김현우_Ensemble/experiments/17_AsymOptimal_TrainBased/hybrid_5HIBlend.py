"""17_hybrid_5HIBlend — per-bearing best selection.

LOBO 분석으로 각 베어링에서 어떤 모델이 강한지 식별 + Test에 적용.

가설:
  - Test1,2 (HI 0.46/0.50): 5_HIBlend mid-life prediction은 train p25 근처로 합리적.
    그러나 17의 train HI-band median은 같은 값으로 수렴 (둘 다 ~32~46k)
  - Test3 (HI 0.16): 5_HIBlend 6449s vs 17 45600s (괴리 매우 큼)
  - Test4 (HI 0.45): 5_HIBlend 14113s vs 17 43800s (괴리 큼)
  - Test5 (HI 0.94): 5_HIBlend 644s vs 17 5400s (괴리 큼)
  - Test6 (HI 0.41): 5_HIBlend 11641s vs 17 28200s (괴리 큼)

per-bearing 결정 로직 (LOBO + train-based 통계 기반):
  Test1,2: Train HI 0.30-0.60 band p25=34200s 부근. 5_HIBlend(32k, 33k) 신뢰.
  Test3: HI=0.16 → Train HI<0.30 분포 median 67212s. 5_HIBlend의 6449s는 분포 밖. 17의 45600s 합리.
  Test4: HI=0.45 → Train 0.30-0.60 median 44388s. 5_HIBlend 14113s, 17 43800s.
        평균 q40=39840 더 합리.
  Test5: HI=0.94 → Train >=0.85 분포 median 5400s. 17의 5400s 정확.
  Test6: HI=0.41 → Train 0.30-0.60 median 44388s. 5_HIBlend 11641s 짧음. 17 28200s 중간.

각 베어링별 train-based "그럴듯한 RUL"을 정량 측정:
  같은 HI band 측정들의 비대칭 페널티 weighted mean asym_score for 후보값.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/hybrid_decision.csv
  artifacts/results/17_AsymOptimal_TrainBased/hybrid_submission.xlsx
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
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


def evaluate_candidate_on_distribution(pred: float, ruls: np.ndarray) -> float:
    """Train rul 분포에서 단일 pred의 평균 asym_score."""
    return float(asym_score([pred] * len(ruls), ruls))


def main() -> None:
    print("=" * 70)
    print("17_hybrid_5HIBlend — per-bearing best selection")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)

    # Load 5_HIBlend predictions
    anchor_csv = RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv"
    anchor = pd.read_csv(anchor_csv)[["Bearing", "RUL_blend_combined_s"]]
    anchor.columns = ["bearing", "anchor_5_HIBlend"]

    # Load 17 predictions
    pred17_csv = RESULT_DIR / "17_test.csv"
    pred17 = pd.read_csv(pred17_csv)[["Bearing", "asym_optimal_pred_s", "expected_score",
                                       "nn_rul_median", "nn_rul_p25", "nn_rul_p40", "nn_rul_p75"]]
    pred17.columns = ["bearing", "pred_17_asym", "exp_score_17",
                       "nn_median", "nn_p25", "nn_p40", "nn_p75"]

    # Load 28_EOL conservative
    eol_csv = RESULT_ROOT / "28_EOLRegressor_Specialist" / "28_eol_regressor_test.csv"
    eol = pd.read_csv(eol_csv)[["Bearing", "EOL_conservative_s", "EOL_median_s"]]
    eol.columns = ["bearing", "pred_28_eol_cons", "pred_28_eol_med"]

    # Last measurement HI from features
    test_last = df[df.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).reset_index(drop=True)
    test_last_simple = test_last[["bearing", "HI", "HI_slope5", "HI_d5",
                                    "rms_multi", "energy_ratio", "chsym_max_env_kurt"]]

    merged = test_last_simple.merge(anchor, on="bearing").merge(pred17, on="bearing").merge(eol, on="bearing")

    # For each test bearing, evaluate ALL candidates against the train HI-band distribution
    # using weighted asym_score (assuming train distribution is the prior).
    final_rows = []
    for _, row in merged.iterrows():
        bearing = row["bearing"]
        hi = float(row["HI"])

        # HI band selection
        if hi < 0.30:
            lo, hi_band = 0.0, 0.30
        elif hi < 0.60:
            lo, hi_band = 0.30, 0.60
        elif hi < 0.85:
            lo, hi_band = 0.60, 0.85
        else:
            lo, hi_band = 0.85, 1.00
        band = train[(train.HI >= lo) & (train.HI < hi_band)]
        ruls_band = band["rul_s"].values

        # Candidates
        candidates = {
            "anchor_5_HIBlend": float(row["anchor_5_HIBlend"]),
            "pred_17_asym": float(row["pred_17_asym"]),
            "pred_28_eol_cons": float(row["pred_28_eol_cons"]),
            "nn_median": float(row["nn_median"]),
            "nn_p25": float(row["nn_p25"]),
            "nn_p40": float(row["nn_p40"]),
        }
        # Evaluate each candidate against HI-band distribution
        cand_scores = {k: evaluate_candidate_on_distribution(v, ruls_band) for k, v in candidates.items()}

        # Best by HI-band score
        best_name = max(cand_scores, key=cand_scores.get)
        best_pred = candidates[best_name]
        best_score = cand_scores[best_name]

        final_rows.append({
            "bearing": bearing, "HI": hi,
            "HI_band": f"[{lo:.2f}, {hi_band:.2f})",
            "band_n": len(band),
            "band_median": float(np.median(ruls_band)) if len(ruls_band) > 0 else np.nan,
            "band_p25": float(np.percentile(ruls_band, 25)) if len(ruls_band) > 0 else np.nan,
            "band_p75": float(np.percentile(ruls_band, 75)) if len(ruls_band) > 0 else np.nan,
            **{k: v for k, v in candidates.items()},
            **{f"exp_{k}": cand_scores[k] for k in candidates},
            "hybrid_best_method": best_name,
            "hybrid_pred_s": best_pred,
            "hybrid_score_expected": best_score,
        })

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(RESULT_DIR / "hybrid_decision.csv", index=False)

    # Print summary
    print(f"  Per-bearing hybrid decisions:\n")
    show_cols = ["bearing", "HI", "HI_band", "band_median", "band_p25", "band_p75",
                 "anchor_5_HIBlend", "pred_17_asym", "pred_28_eol_cons",
                 "hybrid_best_method", "hybrid_pred_s", "hybrid_score_expected"]
    print(out_df[show_cols].to_string(index=False))

    # Submission
    sub = out_df[["bearing", "HI", "hybrid_pred_s"]].copy()
    sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "hybrid_submission.xlsx", index=False)

    # Aggregate
    print(f"\n  Expected mean score (HI-band prior): {out_df['hybrid_score_expected'].mean():.4f}")
    print(f"  Worst-case bearing: {out_df['hybrid_score_expected'].min():.4f}")
    print(f"\n  Saved: {RESULT_DIR / 'hybrid_decision.csv'}")
    print(f"  Saved: {RESULT_DIR / 'hybrid_submission.xlsx'}")


if __name__ == "__main__":
    main()
