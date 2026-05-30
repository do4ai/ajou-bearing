"""17_K_robustness — KNN K hyperparameter sensitivity.

LOBO 4-fold last 1 measurement은 라벨이 600s에 over-fit. Test에서는 다른 분포.
이 script는 K∈{10, 15, 20, 30, 50, 80}를 비교해서:
  - Test prediction 안정성 (variance across K)
  - LOBO expected_score (train distribution 가정 하)
  - 베어링별 prediction range

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/k_robustness.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"

MATCH_FEATURES = [
    "HI", "HI_slope5", "HI_d5", "HI_roll_std5",
    "rms_multi", "energy_ratio", "chsym_max_env_kurt", "chsym_max_kurt",
]
K_VALUES = [10, 15, 20, 30, 50, 80]


def asym_optimal_prediction(rul_samples: np.ndarray, weights: np.ndarray,
                              min_pred: float = 600.0) -> tuple[float, float]:
    max_pred = float(rul_samples.max() * 1.2 + 1000)
    weights = weights / (weights.sum() + 1e-12)
    def neg_obj(p: float) -> float:
        er = 100.0 * (rul_samples - p) / (rul_samples + 1e-12)
        ln_half = np.log(0.5)
        arg_late = np.clip(-ln_half * er / 20.0, -50, 0)
        arg_early = np.clip(ln_half * er / 50.0, -50, 0)
        a = np.where(er <= 0, np.exp(arg_late), np.exp(arg_early))
        return -float(np.sum(weights * a))
    res = minimize_scalar(neg_obj, bounds=(min_pred, max_pred), method="bounded",
                           options={"xatol": 1.0})
    return float(res.x), float(-res.fun)


def main() -> None:
    print("=" * 70)
    print("17_K_robustness — K hyperparameter sensitivity")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    test_last = df[df.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).reset_index(drop=True)

    sc = StandardScaler().fit(train[MATCH_FEATURES].fillna(0).values)
    X_train = sc.transform(train[MATCH_FEATURES].fillna(0).values)
    X_test = sc.transform(test_last[MATCH_FEATURES].fillna(0).values)
    dist = pairwise_distances(X_test, X_train)

    rows = []
    for i, (_, t_row) in enumerate(test_last.iterrows()):
        bearing = t_row["bearing"]
        hi = float(t_row["HI"])
        for k in K_VALUES:
            order = np.argsort(dist[i])[:k]
            ruls = train.iloc[order]["rul_s"].values
            d = dist[i, order]
            w = 1.0 / (d + 1e-6)
            pred_asym, exp_score = asym_optimal_prediction(ruls, w)
            rows.append({
                "bearing": bearing, "HI": hi, "K": k,
                "asym_pred_s": pred_asym, "expected_score": exp_score,
                "nn_median": float(np.median(ruls)),
                "nn_p25": float(np.percentile(ruls, 25)),
                "nn_p40": float(np.percentile(ruls, 40)),
                "nn_p75": float(np.percentile(ruls, 75)),
                "nn_mean": float(ruls.mean()),
                "nn_std": float(ruls.std()),
                "nn_min": float(ruls.min()),
                "nn_max": float(ruls.max()),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULT_DIR / "k_robustness.csv", index=False)
    print(f"  Saved: {RESULT_DIR / 'k_robustness.csv'}")

    # Pivot: K x bearing for asym_pred
    pivot_pred = out_df.pivot(index="bearing", columns="K", values="asym_pred_s")
    pivot_exp = out_df.pivot(index="bearing", columns="K", values="expected_score")
    print("\n  Asym-optimal prediction across K:")
    print(pivot_pred.round(0).to_string())
    print("\n  Expected score across K:")
    print(pivot_exp.round(3).to_string())

    # Stability metrics
    print("\n  Per-bearing K-stability (Coefficient of Variation):")
    for bearing in test_last["bearing"]:
        preds = out_df[out_df.bearing == bearing]["asym_pred_s"].values
        cv = float(preds.std() / preds.mean()) if preds.mean() > 0 else 0.0
        median_pred = float(np.median(preds))
        print(f"    {bearing}: pred_cv={cv:.3f}  median_across_K={median_pred:.0f}s")


if __name__ == "__main__":
    main()
