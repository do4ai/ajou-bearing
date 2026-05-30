"""24_ValidateSelectionMethod — per-bearing 선택 방법 자체를 LOBO로 검증.

핵심 리스크: 1순위(18_PerBearing_Robust)는 'HI-band sensitivity 최대 후보 선택'.
sensitivity grid는 train HI-band prior 가정. Test 진짜 RUL이 이 가정을 따르지
않으면 실패. 1순위는 sensitivity 0.488이지만 LOBO 미검증.

검증 질문: '각 train 베어링을 held-out으로 두고, 동일 per-bearing 선택 로직을
적용했을 때, 그 베어링의 실제 RUL을 5_HIBlend보다 잘 맞히는가?'

방법:
  - 각 train 베어링의 여러 시점(t)에서 그 시점까지만 관측했다고 가정.
  - 그 시점 HI로 HI-band 결정 → 후보들 중 sensitivity 최대 선택.
  - 그 시점 실제 rul_s로 asym_score 평가.
  - 동일 시점에서 5_HIBlend (의 train fold 예측)와 비교.

단, 5_HIBlend의 임의 시점 OOF 예측이 저장되어 있지 않으므로,
여기서는 train-based 후보 (17/19/28 계열의 train OOF)만으로 선택 방법을
검증한다. 즉 "selection이 단일 후보보다 나은가"를 본다.

평가 시점: 각 train 베어링의 t ∈ {25%, 50%, 75%, 90%} progression 지점.
이 지점들의 실제 rul_s는 다양 (600s 라벨 편향 없음).

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/24_selection_validation.csv
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

MATCH_FEATURES = ["HI", "HI_slope5", "HI_d5", "HI_roll_std5",
                  "rms_multi", "energy_ratio", "chsym_max_env_kurt", "chsym_max_kurt"]
K = 20
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


def asym_opt(ruls, weights, lo=600.0):
    ruls = np.asarray(ruls, float); weights = np.asarray(weights, float)
    weights = weights / (weights.sum() + 1e-12)
    hi = float(ruls.max() * 1.2 + 1000)
    def neg(p):
        er = 100.0 * (ruls - p) / (ruls + 1e-12)
        lh = np.log(0.5)
        a = np.where(er <= 0, np.exp(np.clip(-lh*er/20, -50, 0)), np.exp(np.clip(lh*er/50, -50, 0)))
        return -float(np.sum(weights * a))
    return float(minimize_scalar(neg, bounds=(lo, hi), method="bounded").x)


def candidate_preds(ref, query_row, train_full):
    """주어진 query 시점에서 train-based 후보 예측들 생성 (ref = held-out 제외 train)."""
    cols = MATCH_FEATURES
    sc = StandardScaler().fit(ref[cols].fillna(0).values)
    x_ref = sc.transform(ref[cols].fillna(0).values)
    x_q = sc.transform(query_row[cols].fillna(0).values.reshape(1, -1))
    d = pairwise_distances(x_q, x_ref)[0]
    order = np.argsort(d)[:K]
    ruls = ref.iloc[order]["rul_s"].values.astype(float)
    w = 1.0 / (d[order] + 1e-6)
    cands = {
        "knn_asym": asym_opt(ruls, w),
        "knn_q35": float(np.percentile(ruls, 35)),
        "knn_median": float(np.median(ruls)),
        "knn_q25": float(np.percentile(ruls, 25)),
    }
    return cands


def main() -> None:
    print("=" * 70)
    print("24_ValidateSelectionMethod — per-bearing 선택을 LOBO로 검증")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)

    rows = []
    for val in TRAIN_NAMES:
        ref = train[train.bearing != val].reset_index(drop=True)
        vsub = train[train.bearing == val].sort_values("t_s").reset_index(drop=True)
        n = len(vsub)
        # 평가 시점: 25/50/75/90% progression
        for frac in [0.25, 0.50, 0.75, 0.90]:
            idx = min(int(n * frac), n - 1)
            qr = vsub.iloc[idx]
            true_rul = float(qr["rul_s"])
            hi = float(qr["HI"]); band = get_band(hi)
            cands = candidate_preds(ref, qr, train)
            # selection: HI-band sensitivity 최대 후보
            grid = HI_BAND_GRIDS[band]
            cand_sens = {k: float(np.mean([asym_score([v], [g]) for g in grid]))
                         for k, v in cands.items()}
            best = max(cand_sens, key=cand_sens.get)
            sel_pred = cands[best]
            # 실제 성능
            sel_score = asym_score([sel_pred], [true_rul])
            # 비교: 각 단일 후보의 실제 성능
            single_scores = {k: asym_score([v], [true_rul]) for k, v in cands.items()}
            rows.append({
                "val": val, "frac": frac, "idx": idx, "true_rul": true_rul,
                "HI": hi, "band": band,
                "selected": best, "sel_pred": sel_pred, "sel_score": sel_score,
                "best_single_score": max(single_scores.values()),
                "knn_median_score": single_scores["knn_median"],
                "knn_q35_score": single_scores["knn_q35"],
            })

    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "24_selection_validation.csv", index=False)

    print("\n  Selection method vs single candidates (LOBO, 16 평가점):")
    print(f"    selection(HI-band sens) mean asym: {res['sel_score'].mean():.4f}")
    print(f"    oracle best-single        mean asym: {res['best_single_score'].mean():.4f}")
    print(f"    knn_median (단순)          mean asym: {res['knn_median_score'].mean():.4f}")
    print(f"    knn_q35 (보수)             mean asym: {res['knn_q35_score'].mean():.4f}")
    print(f"\n  selection이 knn_median 대비: {res['sel_score'].mean() - res['knn_median_score'].mean():+.4f}")
    print(f"  selection이 oracle 대비 gap: {res['sel_score'].mean() - res['best_single_score'].mean():+.4f}")

    # per-band breakdown
    print("\n  Band별 selection 성능:")
    for band in ["low", "midlow", "midhigh", "high"]:
        sub = res[res.band == band]
        if len(sub):
            print(f"    {band:8s}: n={len(sub)}, sel={sub['sel_score'].mean():.3f}, "
                  f"median={sub['knn_median_score'].mean():.3f}, q35={sub['knn_q35_score'].mean():.3f}")

    # 결론
    sel = res['sel_score'].mean()
    med = res['knn_median_score'].mean()
    q35 = res['knn_q35_score'].mean()
    print("\n  [결론]")
    if sel >= max(med, q35) - 0.01:
        print(f"  → selection 방법이 단일 후보 이상 ({sel:.3f}). per-bearing 선택 정당화.")
    else:
        best_simple = "median" if med > q35 else "q35"
        print(f"  → 단순 {best_simple}({max(med,q35):.3f})가 selection({sel:.3f})보다 나음. "
              f"selection이 HI-band prior에 overfit 가능 → 단순 후보 검토 권장.")
    print(f"\n  Saved: {RESULT_DIR / '24_selection_validation.csv'}")


if __name__ == "__main__":
    main()
