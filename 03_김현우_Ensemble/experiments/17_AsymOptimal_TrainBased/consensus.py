"""20_Consensus — 모든 train-based candidate의 robust 합의.

각 Test 베어링에서 train-based 모델들이 출력한 다양한 RUL 예측의 합의값.
임의 가중치 X → 통계 기반 robust statistic (median, asym-optimal).

Candidates (모두 train-based):
  - 5_HIBlend_combined: LOBO 검증된 anchor (HI-conditioned blend)
  - 17_asym: KNN + asym-optimal
  - 17_hybrid: per-bearing HI-band best
  - 19_robust_asym: HI trajectory + EOL bound cap
  - 19_robust_median: HI trajectory + median
  - 28_eol_cons: EOL specialist conservative
  - 28_eol_med: EOL specialist median

Aggregation:
  - consensus_median: 안정성 우선
  - consensus_asym_optimal: 비대칭 페널티 직접 최적화
  - consensus_trimmed_mean: outlier 제거 후 mean

추가: LOBO + sensitivity joint score
  joint_score = 0.5 * LOBO_score + 0.5 * sensitivity_robust_mean
  → 두 metric 모두 robust한 candidate가 우선

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/20_consensus.csv
  artifacts/results/17_AsymOptimal_TrainBased/20_consensus_submission.xlsx
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
from shared.utils import VAL_NAMES  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")


def asym_optimal_from_samples(rul_samples, min_pred=600.0):
    rul_samples = np.asarray(rul_samples, dtype=np.float64)
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


def trimmed_mean(x, trim_pct=0.20):
    x = np.sort(x)
    n = len(x)
    k = int(np.ceil(n * trim_pct))
    if 2 * k >= n: return float(np.median(x))
    return float(x[k:n-k].mean())


def main() -> None:
    print("=" * 70)
    print("20_Consensus — train-based candidate 합의")
    print("=" * 70)

    # Candidate columns from sensitivity
    sens = pd.read_csv(RESULT_DIR / "sensitivity_matrix.csv")
    candidate_names = ["5_HIBlend_combined", "17_asym", "17_hybrid",
                        "19_robust_asym", "19_robust_median",
                        "28_eol_cons", "28_eol_med"]

    rows = []
    for _, row in sens.iterrows():
        bearing = row["bearing"]
        hi = float(row["HI"])
        preds = np.array([float(row[f"{c}_pred"]) for c in candidate_names])
        sens_means = np.array([float(row[f"{c}_mean"]) for c in candidate_names])

        # Robust aggregation
        median_pred = float(np.median(preds))
        asym_pred = asym_optimal_from_samples(preds)
        tmean_pred = trimmed_mean(preds, 0.20)
        # Weighted by sensitivity score
        w = sens_means / (sens_means.sum() + 1e-12)
        weighted_pred = float(np.sum(w * preds))
        # asym-optimal from candidates with weights = sens_means
        max_pred = float(preds.max() * 1.2 + 1000)
        def neg_obj(p):
            er = 100.0 * (preds - p) / (preds + 1e-12)
            ln_half = np.log(0.5)
            arg_late = np.clip(-ln_half * er / 20.0, -50, 0)
            arg_early = np.clip(ln_half * er / 50.0, -50, 0)
            a = np.where(er <= 0, np.exp(arg_late), np.exp(arg_early))
            return -float(np.sum(w * a))
        res = minimize_scalar(neg_obj, bounds=(600.0, max_pred), method="bounded")
        asym_weighted_pred = float(res.x)

        rec = {
            "bearing": bearing, "HI": hi, "band": row["band"],
            "min_pred": float(preds.min()),
            "max_pred": float(preds.max()),
            "median_pred": median_pred,
            "asym_optimal_pred": asym_pred,
            "trimmed_mean_pred": tmean_pred,
            "weighted_mean_pred": weighted_pred,
            "asym_weighted_pred": asym_weighted_pred,
            "n_candidates": len(preds),
            "candidates_str": ",".join(f"{c}={p:.0f}" for c, p in zip(candidate_names, preds)),
        }
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df["_order"] = out_df["bearing"].map({n: i for i, n in enumerate(VAL_NAMES)})
    out_df = out_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    print(f"\n  Consensus predictions (per Test bearing):\n")
    show_cols = ["bearing", "HI", "min_pred", "max_pred", "median_pred",
                 "asym_optimal_pred", "trimmed_mean_pred", "asym_weighted_pred"]
    print(out_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.0f}"))

    out_df.to_csv(RESULT_DIR / "20_consensus.csv", index=False)

    # Submissions
    for col, fname in [
        ("median_pred", "20_consensus_median_submission.xlsx"),
        ("asym_optimal_pred", "20_consensus_asym_submission.xlsx"),
        ("trimmed_mean_pred", "20_consensus_trimmed_submission.xlsx"),
        ("asym_weighted_pred", "20_consensus_asymweighted_submission.xlsx"),
    ]:
        sub = out_df[["bearing", "HI", col]].copy()
        sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
        sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
        sub.to_excel(RESULT_DIR / fname, index=False)
        print(f"  Saved: {fname}")

    # Sensitivity expected score for each consensus method (cross-validation)
    print("\n  Sensitivity expected score per consensus method (HI-band prior):")
    for col in ["median_pred", "asym_optimal_pred", "trimmed_mean_pred", "asym_weighted_pred"]:
        # Compute sens mean for each consensus pred against same grid
        from shared.utils import asym_score
        HI_BAND_GRIDS = {
            "low": [30000, 50000, 70000, 90000],
            "midlow": [10000, 20000, 30000, 45000, 60000],
            "midhigh": [3000, 5000, 10000, 20000],
            "high": [600, 1500, 3000, 5000, 9000, 15000],
        }
        scores = []
        for _, row in out_df.iterrows():
            grid = HI_BAND_GRIDS[row["band"]]
            pred = float(row[col])
            s = float(np.mean([asym_score([pred], [g]) for g in grid]))
            scores.append(s)
        print(f"    {col}: mean={np.mean(scores):.4f}  worst={min(scores):.4f}")


if __name__ == "__main__":
    main()
