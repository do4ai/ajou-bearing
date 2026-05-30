"""17_sensitivity — Test 진짜 RUL 가정 grid x 모든 candidate 점수.

Test 진짜 RUL을 알 수 없으므로 plausible 후보 RUL 시나리오에 대해 각 candidate의
asym_score를 계산하여 robust한 submission 결정.

Train-based scenarios (per HI band):
  HI < 0.3:     true ∈ {30k, 50k, 70k, 90k}s
  HI 0.3-0.6:   true ∈ {10k, 20k, 30k, 45k, 60k}s
  HI 0.6-0.85:  true ∈ {3k, 5k, 10k, 20k}s
  HI >= 0.85:   true ∈ {600, 1.5k, 3k, 5k, 9k, 15k}s

각 시나리오 등확률 가정 시 mean asym_score = robust score.
worst-case scenario asym_score = worst-case robustness.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/sensitivity_matrix.csv
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
from shared.utils import asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")

# Train-based plausible RUL grids per HI band (시간 단위 sec)
HI_BAND_GRIDS = {
    "low": [30000, 50000, 70000, 90000],          # HI < 0.3 (Train HI<0.3 median 67200, IQR 60300-73800)
    "midlow": [10000, 20000, 30000, 45000, 60000], # HI 0.3-0.6 (Train median 44400, IQR 34200-54000)
    "midhigh": [3000, 5000, 10000, 20000],         # HI 0.6-0.85 (Train median 20988, IQR 15000-27600)
    "high": [600, 1500, 3000, 5000, 9000, 15000],  # HI 0.85+ (Train median 5400, IQR 2988-9000)
}


def get_band(hi: float) -> str:
    if hi < 0.30: return "low"
    if hi < 0.60: return "midlow"
    if hi < 0.85: return "midhigh"
    return "high"


def main() -> None:
    print("=" * 70)
    print("17_sensitivity — Test 진짜 RUL 가정 grid")
    print("=" * 70)

    # Load all candidate predictions
    anchor = pd.read_csv(RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv")
    anchor = anchor[["Bearing", "RUL_blend_combined_s", "RUL_blend_full_s"]]
    anchor.columns = ["bearing", "5_HIBlend_combined", "5_HIBlend_full"]

    sa16 = pd.read_csv(RESULT_ROOT / "16_ScoreAware_CalibratedEnsemble" / "16_scoreaware_debug.csv")
    sa16 = sa16[["bearing", "balanced_rul_s", "safe_rul_s", "aggressive_rul_s"]]
    sa16.columns = ["bearing", "16_balanced", "16_safe", "16_aggressive"]

    p17 = pd.read_csv(RESULT_DIR / "17_test.csv")[["Bearing", "asym_optimal_pred_s", "HI_last"]]
    p17.columns = ["bearing", "17_asym", "HI"]

    p17h = pd.read_csv(RESULT_DIR / "hybrid_decision.csv")[["bearing", "hybrid_pred_s"]]
    p17h.columns = ["bearing", "17_hybrid"]

    p28 = pd.read_csv(RESULT_ROOT / "28_EOLRegressor_Specialist" / "28_eol_regressor_test.csv")
    p28 = p28[["Bearing", "EOL_conservative_s", "EOL_median_s"]]
    p28.columns = ["bearing", "28_eol_cons", "28_eol_med"]

    # 19_EOLProgression_Robust (LOBO 0.75!)
    p19r = pd.read_csv(RESULT_DIR / "19_eol_progression_robust.csv")
    p19r = p19r[["Bearing", "asym_optimal_pred_s", "median_pred_s"]]
    p19r.columns = ["bearing", "19_robust_asym", "19_robust_median"]

    merged = p17.merge(anchor, on="bearing").merge(sa16, on="bearing") \
                .merge(p17h, on="bearing").merge(p28, on="bearing") \
                .merge(p19r, on="bearing")

    candidates = ["5_HIBlend_combined", "5_HIBlend_full", "16_balanced", "16_safe",
                  "16_aggressive", "17_asym", "17_hybrid",
                  "28_eol_cons", "28_eol_med",
                  "19_robust_asym", "19_robust_median"]

    # Sensitivity matrix: per-bearing, per-candidate, mean score over scenario grid
    rows = []
    for _, row in merged.iterrows():
        bearing = row["bearing"]
        hi = float(row["HI"])
        band = get_band(hi)
        grid = HI_BAND_GRIDS[band]
        rec = {"bearing": bearing, "HI": hi, "band": band, "grid": str(grid)}
        for cand in candidates:
            pred = float(row[cand])
            scores = [asym_score([pred], [g]) for g in grid]
            rec[f"{cand}_pred"] = pred
            rec[f"{cand}_mean"] = float(np.mean(scores))
            rec[f"{cand}_min"] = float(np.min(scores))
            rec[f"{cand}_worst_rul"] = float(grid[int(np.argmin(scores))])
        rows.append(rec)

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(RESULT_DIR / "sensitivity_matrix.csv", index=False)

    # Aggregate per candidate: mean of per-bearing mean (overall robust score)
    print(f"\n  Candidate rankings (6 베어링 평균 / worst-bearing 평균):\n")
    agg_rows = []
    for cand in candidates:
        m_mean = sens_df[f"{cand}_mean"].mean()
        m_min = sens_df[f"{cand}_min"].mean()
        worst_b = sens_df[f"{cand}_mean"].min()  # worst bearing's mean score
        agg_rows.append({"candidate": cand,
                          "robust_mean_score": m_mean,
                          "worst_case_avg": m_min,
                          "worst_bearing_mean": worst_b})
    agg = pd.DataFrame(agg_rows).sort_values("robust_mean_score", ascending=False)
    print(agg.to_string(index=False))

    agg.to_csv(RESULT_DIR / "sensitivity_aggregate.csv", index=False)

    print(f"\n  Per-bearing detail (best candidate):")
    for _, row in sens_df.iterrows():
        means = {c: row[f"{c}_mean"] for c in candidates}
        best_c = max(means, key=means.get)
        print(f"    {row['bearing']} (HI={row['HI']:.3f}, band={row['band']}): "
              f"best={best_c}={row[f'{best_c}_pred']:.0f}s, "
              f"robust_mean={row[f'{best_c}_mean']:.3f}, worst={row[f'{best_c}_min']:.3f}")

    print(f"\n  Saved: {RESULT_DIR / 'sensitivity_matrix.csv'}")
    print(f"  Saved: {RESULT_DIR / 'sensitivity_aggregate.csv'}")


if __name__ == "__main__":
    main()
