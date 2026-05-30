"""17_AsymOptimal_TrainBased — HI band Train RUL 분포 + 비대칭 최적 단일 예측.

핵심 원칙:
  - 완전 train-based: Test와 가장 닮은 train 측정들의 실제 rul_s 분포 사용
  - 비대칭 페널티 직접 최적화: argmax_p E_R[asym_score(p, R)]
  - 임의 숫자 박기 없음. 600s 물리 하한만.

방법:
  1. Train의 (HI, HI_slope5, energy_ratio) 3D feature space에서 Test와 가장 가까운 K=30 train 측정
  2. 그 K개 측정의 실제 rul_s 수집 → empirical distribution P(R | Test state)
  3. scipy.optimize: optimal_p = argmax_p sum_{r in P} asym_score(p, r)
  4. LOBO 검증: train의 한 베어링을 빼고 같은 방식으로 holdout last asym_score 계산

이는 K-NN regression의 한 형태이지만:
  - 단순 mean/median이 아닌 비대칭 페널티 최적화 quantile
  - 평가 metric이 학습 objective와 정확히 일치 (정합성)

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/17_lobo.csv
  artifacts/results/17_AsymOptimal_TrainBased/17_test.csv
  artifacts/results/17_AsymOptimal_TrainBased/17_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"

# State match features (Train과 Test의 "현재 상태" 비교에 사용)
# HI는 핵심, slope/d로 추세 반영, energy_ratio/chsym_max_env_kurt로 진동 패턴 보강.
MATCH_FEATURES = [
    "HI", "HI_slope5", "HI_d5", "HI_roll_std5",
    "rms_multi", "energy_ratio", "chsym_max_env_kurt", "chsym_max_kurt",
]
K_NEIGHBORS = 30


def asym_score_single(pred: float, true: float) -> float:
    """Single sample asym_score (vectorized 가능하지만 명확성 위해 단일)."""
    if true <= 0:
        return 0.0
    er = 100.0 * (true - pred) / true
    ln_half = np.log(0.5)
    if er <= 0:
        return float(np.exp(np.clip(-ln_half * er / 20.0, -50, 0)))
    else:
        return float(np.exp(np.clip(ln_half * er / 50.0, -50, 0)))


def asym_optimal_prediction(rul_samples: np.ndarray, weights: np.ndarray = None,
                              min_pred: float = 600.0, max_pred: float = None) -> tuple[float, float]:
    """주어진 RUL empirical distribution에서 비대칭 페널티 기대값 최대화하는 단일 prediction.

    objective: argmax_p sum_i w_i * asym_score(p, r_i)
    """
    rul_samples = np.asarray(rul_samples, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(rul_samples) / len(rul_samples)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

    if max_pred is None:
        max_pred = float(rul_samples.max() * 1.2 + 1000)

    def neg_obj(p: float) -> float:
        # vectorized asym_score
        er = 100.0 * (rul_samples - p) / (rul_samples + 1e-12)
        ln_half = np.log(0.5)
        arg_late = np.clip(-ln_half * er / 20.0, -50, 0)
        arg_early = np.clip(ln_half * er / 50.0, -50, 0)
        a = np.where(er <= 0, np.exp(arg_late), np.exp(arg_early))
        return -float(np.sum(weights * a))

    res = minimize_scalar(neg_obj, bounds=(min_pred, max_pred), method="bounded",
                           options={"xatol": 1.0})
    return float(res.x), float(-res.fun)


def knn_state_match(train_ref: pd.DataFrame, test_query: pd.DataFrame,
                     features: list[str], k: int = K_NEIGHBORS) -> list[dict]:
    """train_ref에서 test_query의 각 row에 가장 가까운 k개의 train 측정 찾기."""
    sc = StandardScaler().fit(train_ref[features].fillna(0).values)
    X_ref = sc.transform(train_ref[features].fillna(0).values)
    X_q = sc.transform(test_query[features].fillna(0).values)
    dist = pairwise_distances(X_q, X_ref)
    results = []
    for qi in range(len(test_query)):
        order = np.argsort(dist[qi])[:k]
        nn = train_ref.iloc[order].copy()
        d = dist[qi, order]
        w = 1.0 / (d + 1e-6)
        results.append({
            "neighbors_rul_s": nn["rul_s"].values.astype(np.float64),
            "neighbors_dist": d.astype(np.float64),
            "neighbors_weights": w,
            "neighbors_bearing": nn["bearing"].values,
        })
    return results


def main() -> None:
    print("=" * 70)
    print("17_AsymOptimal_TrainBased — HI-band 분포 + 비대칭 최적 예측")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    print(f"  features used for state match: {MATCH_FEATURES}")
    print(f"  k neighbors: {K_NEIGHBORS}")

    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    test_last = df[df.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).reset_index(drop=True)

    # ── LOBO 검증 (4-fold) ─────────────────────────────────────────
    print("\n[LOBO 4-fold validation]")
    lobo_rows = []
    for val in TRAIN_NAMES:
        ref = train[train.bearing != val].reset_index(drop=True)
        # query: 빠진 베어링의 마지막 측정 1개
        query = train[train.bearing == val].tail(1).reset_index(drop=True)
        nn_info = knn_state_match(ref, query, MATCH_FEATURES, k=K_NEIGHBORS)[0]
        ruls = nn_info["neighbors_rul_s"]
        weights = nn_info["neighbors_weights"]
        # asymmetric-optimal single prediction
        pred_asym, expected_score = asym_optimal_prediction(ruls, weights=weights)
        # baseline candidates for comparison
        pred_median = float(np.median(ruls))
        pred_q25 = float(np.percentile(ruls, 25))
        pred_q40 = float(np.percentile(ruls, 40))
        true = float(query.iloc[0]["rul_s"])

        score_asym = asym_score_single(pred_asym, true)
        score_median = asym_score_single(pred_median, true)
        score_q25 = asym_score_single(pred_q25, true)
        score_q40 = asym_score_single(pred_q40, true)

        lobo_rows.append({
            "val": val, "true_rul_s": true,
            "pred_asym_optimal": pred_asym, "score_asym_optimal": score_asym,
            "pred_median": pred_median, "score_median": score_median,
            "pred_q25": pred_q25, "score_q25": score_q25,
            "pred_q40": pred_q40, "score_q40": score_q40,
            "expected_score_asym": expected_score,
            "nn_rul_min": float(ruls.min()), "nn_rul_max": float(ruls.max()),
            "nn_rul_mean": float(ruls.mean()), "nn_rul_std": float(ruls.std()),
            "nn_bearings": ",".join(sorted(set(nn_info["neighbors_bearing"]))),
        })
        print(f"  Fold {val}: true={true:.0f}  asym_opt={pred_asym:.0f}(score={score_asym:.3f})  "
              f"median={pred_median:.0f}(score={score_median:.3f})  "
              f"q40={pred_q40:.0f}(score={score_q40:.3f})")

    lobo_df = pd.DataFrame(lobo_rows)
    lobo_df.to_csv(RESULT_DIR / "17_lobo.csv", index=False)
    print(f"\n  LOBO averages:")
    print(f"    asym_optimal: {lobo_df['score_asym_optimal'].mean():.4f}")
    print(f"    median:       {lobo_df['score_median'].mean():.4f}")
    print(f"    q25:          {lobo_df['score_q25'].mean():.4f}")
    print(f"    q40:          {lobo_df['score_q40'].mean():.4f}")

    # ── Test 예측 ──────────────────────────────────────────────────
    print("\n[Test inference]")
    nn_info_test = knn_state_match(train, test_last, MATCH_FEATURES, k=K_NEIGHBORS)
    rows_out = []
    for i, (_, row) in enumerate(test_last.iterrows()):
        nn = nn_info_test[i]
        ruls = nn["neighbors_rul_s"]
        weights = nn["neighbors_weights"]
        pred_asym, exp_score = asym_optimal_prediction(ruls, weights=weights)
        pred_median = float(np.median(ruls))
        pred_q25 = float(np.percentile(ruls, 25))
        pred_q40 = float(np.percentile(ruls, 40))
        pred_q50 = float(np.percentile(ruls, 50))
        rows_out.append({
            "Bearing": row["bearing"],
            "HI_last": float(row["HI"]),
            "HI_slope5_last": float(row.get("HI_slope5", 0.0)),
            "energy_ratio_last": float(row.get("energy_ratio", 0.0)),
            "nn_rul_mean": float(ruls.mean()),
            "nn_rul_median": float(np.median(ruls)),
            "nn_rul_p25": pred_q25,
            "nn_rul_p40": pred_q40,
            "nn_rul_p75": float(np.percentile(ruls, 75)),
            "nn_rul_min": float(ruls.min()),
            "nn_rul_max": float(ruls.max()),
            "asym_optimal_pred_s": pred_asym,
            "expected_score": exp_score,
            "nn_bearings": ",".join(sorted(set(nn["neighbors_bearing"]))),
        })
        print(f"  {row['bearing']}: HI={row['HI']:.3f}  asym_opt={pred_asym:.0f}s  "
              f"(median={pred_median:.0f}, q25={pred_q25:.0f}, q40={pred_q40:.0f}, "
              f"nn_range=[{ruls.min():.0f}, {ruls.max():.0f}], exp={exp_score:.3f})")

    test_df = pd.DataFrame(rows_out)
    test_df.to_csv(RESULT_DIR / "17_test.csv", index=False)

    sub = test_df[["Bearing", "HI_last", "asym_optimal_pred_s"]].copy()
    sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "17_submission.xlsx", index=False)

    print(f"\n  Saved: {RESULT_DIR / '17_lobo.csv'}")
    print(f"  Saved: {RESULT_DIR / '17_test.csv'}")
    print(f"  Saved: {RESULT_DIR / '17_submission.xlsx'}")


if __name__ == "__main__":
    main()
