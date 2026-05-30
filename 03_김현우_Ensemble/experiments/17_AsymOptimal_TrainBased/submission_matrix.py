"""21_SubmissionMatrix — 모든 train-based candidate × 6 베어링 통합 테이블.

각 후보의 LOBO score, sensitivity expected score, per-bearing 예측 모두 한 표.
최종 1+1 제출 결정 근거.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/21_submission_matrix.csv
  artifacts/results/17_AsymOptimal_TrainBased/21_submission_summary.csv
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
from shared.utils import VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")

HI_BAND_GRIDS = {
    "low": [30000, 50000, 70000, 90000],
    "midlow": [10000, 20000, 30000, 45000, 60000],
    "midhigh": [3000, 5000, 10000, 20000],
    "high": [600, 1500, 3000, 5000, 9000, 15000],
}


def get_band(hi):
    if hi < 0.30: return "low"
    if hi < 0.60: return "midlow"
    if hi < 0.85: return "midhigh"
    return "high"


def sens_score(pred, band):
    grid = HI_BAND_GRIDS[band]
    return float(np.mean([asym_score([pred], [g]) for g in grid]))


def main() -> None:
    print("=" * 70)
    print("21_SubmissionMatrix — 모든 train-based 후보 통합")
    print("=" * 70)

    # Sensitivity matrix as base
    sens = pd.read_csv(RESULT_DIR / "sensitivity_matrix.csv")

    # All candidate predictions
    candidates = {
        "5_HIBlend_combined": "sensitivity",  # already in sens
        "5_HIBlend_full": "sensitivity",
        "16_balanced": "sensitivity",
        "16_safe": "sensitivity",
        "17_asym": "sensitivity",
        "17_hybrid": "sensitivity",
        "19_robust_asym": "sensitivity",
        "19_robust_median": "sensitivity",
        "28_eol_cons": "sensitivity",
        "28_eol_med": "sensitivity",
    }

    # LOBO scores per candidate (known from iter 1-4 work)
    LOBO_SCORES = {
        "5_HIBlend_combined": 0.712,  # combined LOBO
        "5_HIBlend_full": 0.638,      # full strategy LOBO
        "16_balanced": None,
        "16_safe": None,
        "17_asym": 0.000,             # LOBO last 1
        "17_hybrid": None,
        "17_asym_LOBO_full": 0.520,   # LOBO full
        "19_robust_asym": 0.750,      # robust LOBO mean
        "19_robust_median": 0.750,
        "28_eol_cons": 0.054,
        "28_eol_med": 0.054,
    }

    # Per-bearing matrix
    rows = []
    for _, srow in sens.iterrows():
        bearing = srow["bearing"]
        hi = float(srow["HI"])
        band = srow["band"]
        rec = {"bearing": bearing, "HI": hi, "band": band}
        for cand in candidates:
            pred = float(srow[f"{cand}_pred"])
            sens_mean = float(srow[f"{cand}_mean"])
            rec[f"{cand}_pred"] = pred
            rec[f"{cand}_sens_mean"] = sens_mean
        # Add consensus predictions
        consensus_df = pd.read_csv(RESULT_DIR / "20_consensus.csv")
        crow = consensus_df[consensus_df.bearing == bearing].iloc[0]
        rec["consensus_median"] = float(crow["median_pred"])
        rec["consensus_asym"] = float(crow["asym_optimal_pred"])
        rec["consensus_asym_weighted"] = float(crow["asym_weighted_pred"])
        rec["consensus_trimmed"] = float(crow["trimmed_mean_pred"])
        # Sensitivity for consensus
        rec["consensus_median_sens"] = sens_score(rec["consensus_median"], band)
        rec["consensus_asym_sens"] = sens_score(rec["consensus_asym"], band)
        rec["consensus_asym_weighted_sens"] = sens_score(rec["consensus_asym_weighted"], band)
        rec["consensus_trimmed_sens"] = sens_score(rec["consensus_trimmed"], band)
        # Per-bearing robust (best of all candidates)
        all_sens = {k: rec[f"{k}_sens_mean"] for k in candidates}
        all_sens["consensus_median"] = rec["consensus_median_sens"]
        all_sens["consensus_asym"] = rec["consensus_asym_sens"]
        all_sens["consensus_asym_weighted"] = rec["consensus_asym_weighted_sens"]
        all_sens["consensus_trimmed"] = rec["consensus_trimmed_sens"]
        best = max(all_sens, key=all_sens.get)
        rec["per_bearing_best"] = best
        rec["per_bearing_best_sens"] = all_sens[best]
        if best.startswith("consensus"):
            rec["per_bearing_best_pred"] = rec[best]
        else:
            rec["per_bearing_best_pred"] = rec[f"{best}_pred"]
        rows.append(rec)

    matrix = pd.DataFrame(rows)
    matrix["_order"] = matrix["bearing"].map({n: i for i, n in enumerate(VAL_NAMES)})
    matrix = matrix.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    matrix.to_csv(RESULT_DIR / "21_submission_matrix.csv", index=False)

    # Print per-bearing predictions table
    print("\n  Per-bearing predictions (모든 후보 + consensus + per-bearing best):\n")
    pred_cols = [f"{c}_pred" for c in ["5_HIBlend_combined", "17_asym", "17_hybrid",
                                        "19_robust_asym", "28_eol_cons", "28_eol_med"]]
    show_cols = ["bearing", "HI"] + pred_cols + ["consensus_median", "consensus_asym",
                                                   "per_bearing_best", "per_bearing_best_pred"]
    print(matrix[show_cols].to_string(index=False, float_format=lambda x: f"{x:.0f}" if abs(x) > 10 else f"{x:.2f}"))

    # Summary: mean sensitivity + LOBO per candidate
    print("\n  Candidate summary (mean sensitivity over 6 베어링):")
    summary_rows = []
    for cand in candidates:
        sens_means = [float(matrix.iloc[i][f"{cand}_sens_mean"]) for i in range(len(matrix))]
        summary_rows.append({
            "candidate": cand,
            "sens_mean_6bearing": np.mean(sens_means),
            "sens_worst_bearing": np.min(sens_means),
            "LOBO_known": LOBO_SCORES.get(cand, None),
        })
    for col in ["consensus_median", "consensus_asym", "consensus_asym_weighted", "consensus_trimmed"]:
        col_full = f"{col}_sens"
        sens_means = matrix[col_full].values
        summary_rows.append({
            "candidate": col,
            "sens_mean_6bearing": float(np.mean(sens_means)),
            "sens_worst_bearing": float(np.min(sens_means)),
            "LOBO_known": None,
        })
    # Per-bearing best mix
    per_b_best_sens = matrix["per_bearing_best_sens"].mean()
    summary_rows.append({
        "candidate": "per_bearing_best_mix (Submission18)",
        "sens_mean_6bearing": float(per_b_best_sens),
        "sens_worst_bearing": float(matrix["per_bearing_best_sens"].min()),
        "LOBO_known": None,
    })
    summary = pd.DataFrame(summary_rows).sort_values("sens_mean_6bearing", ascending=False)
    summary.to_csv(RESULT_DIR / "21_submission_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    print(f"\n  Saved: {RESULT_DIR / '21_submission_matrix.csv'}")
    print(f"  Saved: {RESULT_DIR / '21_submission_summary.csv'}")


if __name__ == "__main__":
    main()
