"""23_BetaSweep — conservative tilt β 검증.

비대칭 페널티상 늦은 예측이 2.5배 가혹 → 예측을 약간 짧게(β<1) 미는 것이
robust score를 올릴 수 있는지 HI-band sensitivity grid로 정량 검증.

대상: 1순위 (per_bearing_robust) 예측에 β ∈ {0.80~1.05} 곱.
β는 LOBO/train-based prior로만 평가 (test 라벨 미사용).

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/23_beta_sweep.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")

# HI-band별 train RUL grid (sensitivity prior, 초)
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


def robust_score(pred, band):
    grid = HI_BAND_GRIDS[band]
    return float(np.mean([asym_score([pred], [g]) for g in grid]))


def main() -> None:
    print("=" * 70)
    print("23_BetaSweep — conservative tilt 검증 (1순위 per-bearing)")
    print("=" * 70)

    base = pd.read_csv(RESULT_DIR / "18_per_bearing_robust_debug.csv")

    betas = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 1.00, 1.02, 1.05]
    rows = []
    for beta in betas:
        per_b = []
        for _, r in base.iterrows():
            hi = float(r["HI_last"]); band = get_band(hi)
            pred = max(600.0, float(r["RUL_pred_seconds"]) * beta)
            per_b.append(robust_score(pred, band))
        rows.append({"beta": beta, "robust_mean": float(np.mean(per_b)),
                     "robust_worst": float(np.min(per_b))})
        print(f"  β={beta:.2f}: robust_mean={np.mean(per_b):.4f}  worst={np.min(per_b):.4f}")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULT_DIR / "23_beta_sweep.csv", index=False)

    best = sweep.loc[sweep["robust_mean"].idxmax()]
    base_score = float(sweep[sweep.beta == 1.00]["robust_mean"].iloc[0])
    print(f"\n  β=1.00 baseline robust_mean: {base_score:.4f}")
    print(f"  Best β={best['beta']:.2f} robust_mean={best['robust_mean']:.4f}  "
          f"(Δ={best['robust_mean']-base_score:+.4f})")

    if best["beta"] != 1.00 and best["robust_mean"] - base_score > 0.005:
        print(f"\n  → conservative tilt β={best['beta']:.2f} 유의미 개선. 적용 검토.")
        # Generate tilted submission
        out = base[["Bearing", "HI_last"]].copy()
        out["RUL_pred_seconds"] = np.maximum(600.0, base["RUL_pred_seconds"] * best["beta"])
        out["RUL_pred_hours"] = out["RUL_pred_seconds"] / 3600.0
        out.to_excel(RESULT_DIR / "23_per_bearing_beta_tilted_submission.xlsx", index=False)
        print(f"  Saved tilted: 23_per_bearing_beta_tilted_submission.xlsx")
    else:
        print(f"\n  → β=1.00 (no tilt)이 충분히 robust. 추가 보정 불필요.")

    print(f"\n  Saved: {RESULT_DIR / '23_beta_sweep.csv'}")


if __name__ == "__main__":
    main()
