"""17_LOBO_multipoint — last 1 단일 측정의 한계 극복.

기존 LOBO last 1은 모든 fold true=600s → 17번 KNN이 못 맞춤 → score=0
그러나 Test는 EOL 미도달 → true ≠ 600s
→ multi-point LOBO로 train의 last K개 측정에 대해 평가 (다양한 RUL)

last 5: true ∈ {3000, 2400, 1800, 1200, 600}s
last 10: true ∈ {6000, 5400, ..., 600}s
last 20: true ∈ {12000, ..., 600}s

→ 17번의 진짜 robustness 측정. K=20 사용.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/lobo_multipoint.csv
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
from shared.utils import TRAIN_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"

MATCH_FEATURES = [
    "HI", "HI_slope5", "HI_d5", "HI_roll_std5",
    "rms_multi", "energy_ratio", "chsym_max_env_kurt", "chsym_max_kurt",
]
K_NEIGHBORS = 20
EVAL_LAST_K = [1, 5, 10, 20]


def asym_optimal_prediction(rul_samples, weights, min_pred=600.0):
    max_pred = float(rul_samples.max() * 1.2 + 1000)
    weights = weights / (weights.sum() + 1e-12)
    def neg_obj(p):
        er = 100.0 * (rul_samples - p) / (rul_samples + 1e-12)
        ln_half = np.log(0.5)
        arg_late = np.clip(-ln_half * er / 20.0, -50, 0)
        arg_early = np.clip(ln_half * er / 50.0, -50, 0)
        a = np.where(er <= 0, np.exp(arg_late), np.exp(arg_early))
        return -float(np.sum(weights * a))
    res = minimize_scalar(neg_obj, bounds=(min_pred, max_pred), method="bounded",
                           options={"xatol": 1.0})
    return float(res.x)


def main() -> None:
    print("=" * 70)
    print("17_LOBO_multipoint — last 5/10/20 multi-point evaluation")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)

    sc = StandardScaler().fit(train[MATCH_FEATURES].fillna(0).values)

    all_results = []
    summary_rows = []
    for val in TRAIN_NAMES:
        ref = train[train.bearing != val].reset_index(drop=True)
        query_all = train[train.bearing == val].sort_values("t_s").reset_index(drop=True)

        X_ref = sc.transform(ref[MATCH_FEATURES].fillna(0).values)
        X_q = sc.transform(query_all[MATCH_FEATURES].fillna(0).values)
        dist = pairwise_distances(X_q, X_ref)

        preds = []
        trues = []
        for qi in range(len(query_all)):
            order = np.argsort(dist[qi])[:K_NEIGHBORS]
            ruls = ref.iloc[order]["rul_s"].values
            d = dist[qi, order]
            w = 1.0 / (d + 1e-6)
            pred = asym_optimal_prediction(ruls, w)
            preds.append(pred)
            trues.append(float(query_all.iloc[qi]["rul_s"]))

        preds_arr = np.array(preds)
        trues_arr = np.array(trues)

        # Per-measurement detail
        for qi, (p, t) in enumerate(zip(preds_arr, trues_arr)):
            all_results.append({
                "fold": val, "idx": qi, "t_s": float(query_all.iloc[qi]["t_s"]),
                "true_rul_s": t, "pred_asym": p,
                "HI": float(query_all.iloc[qi]["HI"]),
                "asym_score": asym_score([p], [t]),
            })

        # Summary per evaluation horizon
        row = {"fold": val, "n_total": len(query_all)}
        for k in EVAL_LAST_K:
            ks = min(k, len(query_all))
            p_last_k = preds_arr[-ks:]
            t_last_k = trues_arr[-ks:]
            row[f"asym_last{k}_mean"] = asym_score(p_last_k, t_last_k)
            row[f"asym_last{k}_min"] = float(min(asym_score([pi], [ti]) for pi, ti in zip(p_last_k, t_last_k)))
        # Full timeline
        row["asym_full_mean"] = asym_score(preds_arr, trues_arr)
        summary_rows.append(row)
        print(f"  {val}: n={len(query_all)}  "
              f"last1={row['asym_last1_mean']:.3f}  "
              f"last5={row['asym_last5_mean']:.3f}  "
              f"last10={row['asym_last10_mean']:.3f}  "
              f"last20={row['asym_last20_mean']:.3f}  "
              f"full={row['asym_full_mean']:.3f}")

    pd.DataFrame(all_results).to_csv(RESULT_DIR / "lobo_multipoint_detail.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULT_DIR / "lobo_multipoint.csv", index=False)

    print("\n  4-fold averages:")
    for k in EVAL_LAST_K:
        m = summary_df[f"asym_last{k}_mean"].mean()
        worst = summary_df[f"asym_last{k}_mean"].min()
        print(f"    last{k:2d}: mean={m:.4f}  worst_fold={worst:.4f}")
    print(f"    full  : mean={summary_df['asym_full_mean'].mean():.4f}  worst_fold={summary_df['asym_full_mean'].min():.4f}")
    print(f"\n  Saved: {RESULT_DIR / 'lobo_multipoint.csv'}")


if __name__ == "__main__":
    main()
