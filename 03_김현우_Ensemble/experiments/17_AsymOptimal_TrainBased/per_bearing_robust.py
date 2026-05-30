"""18_PerBearing_Robust — sensitivity grid 기반 베어링별 최적 후보 선택.

각 베어링에서 train-based plausible RUL grid에 대한 mean asym_score 최대 candidate 선택.
Per-bearing optimal mix = sensitivity 평균 0.488 (모든 single 후보 능가).

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/18_per_bearing_robust_submission.xlsx
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
from tools.paths import add_repo_to_path, result_dir
add_repo_to_path()

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")


def main() -> None:
    print("=" * 70)
    print("18_PerBearing_Robust — sensitivity 기반 베어링별 최적 후보 선택")
    print("=" * 70)

    sens = pd.read_csv(RESULT_DIR / "sensitivity_matrix.csv")
    candidates = ["5_HIBlend_combined", "5_HIBlend_full", "16_balanced", "16_safe",
                  "16_aggressive", "17_asym", "17_hybrid", "28_eol_cons", "28_eol_med"]

    rows = []
    for _, row in sens.iterrows():
        means = {c: float(row[f"{c}_mean"]) for c in candidates}
        best_c = max(means, key=means.get)
        pred = float(row[f"{best_c}_pred"])
        rows.append({
            "Bearing": row["bearing"],
            "HI_last": float(row["HI"]),
            "band": row["band"],
            "best_method": best_c,
            "RUL_pred_seconds": pred,
            "RUL_pred_hours": pred / 3600.0,
            "robust_mean_score": means[best_c],
            "worst_case_score": float(row[f"{best_c}_min"]),
        })

    out_df = pd.DataFrame(rows).sort_values("Bearing")
    print("\n  Per-bearing selection:")
    print(out_df.to_string(index=False))
    print(f"\n  Mean robust score: {out_df['robust_mean_score'].mean():.4f}")
    print(f"  Mean worst-case:   {out_df['worst_case_score'].mean():.4f}")

    sub = out_df[["Bearing", "HI_last", "RUL_pred_seconds", "RUL_pred_hours"]].copy()
    sub.to_excel(RESULT_DIR / "18_per_bearing_robust_submission.xlsx", index=False)
    out_df.to_csv(RESULT_DIR / "18_per_bearing_robust_debug.csv", index=False)
    print(f"\n  Saved: {RESULT_DIR / '18_per_bearing_robust_submission.xlsx'}")


if __name__ == "__main__":
    main()
