"""26_SelectionRuleSearch — 더 나은 deterministic 선택 규칙 탐색 (LOBO unbiased).

배경: 24_ValidateSelectionMethod에서 현 HI-band sensitivity 선택 = 0.519,
oracle best-single = 0.667 (gap 0.148). 이 gap을 닫을 수 있는,
overfit 아닌(0~1 param, LOBO out-of-bearing 검증) 규칙이 있는지 탐색.

후보 규칙 (모두 동일 16 평가점에서 비교):
  - fixed: 항상 knn_asym / q35 / median / q25 중 하나
  - blend: (asym+q35)/2, (q25+median)/2 등 고정 가중
  - band_sens: 현 1순위 규칙 (HI-band sensitivity 최대)
  - grid_asymopt: KNN 분포 + band grid 를 합친 분포의 asym-opt
  - shrink: knn_asym 을 band median 쪽으로 lambda 만큼 수축 (lambda LOBO sweep)

결정 기준: 어떤 단순 규칙이 band_sens(0.519) 대비 +0.02 이상이면 채택 후보.
출력: artifacts/results/17_AsymOptimal_TrainBased/26_selection_rule_search.csv
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


def knn_dist(ref, query_row):
    cols = MATCH_FEATURES
    sc = StandardScaler().fit(ref[cols].fillna(0).values)
    x_ref = sc.transform(ref[cols].fillna(0).values)
    x_q = sc.transform(query_row[cols].fillna(0).values.reshape(1, -1))
    d = pairwise_distances(x_q, x_ref)[0]
    order = np.argsort(d)[:K]
    ruls = ref.iloc[order]["rul_s"].values.astype(float)
    w = 1.0 / (d[order] + 1e-6)
    return ruls, w


def candidates(ruls, w):
    return {
        "asym": asym_opt(ruls, w),
        "q35": float(np.percentile(ruls, 35)),
        "median": float(np.median(ruls)),
        "q25": float(np.percentile(ruls, 25)),
    }


def main() -> None:
    print("=" * 70)
    print("26_SelectionRuleSearch — 더 나은 선택 규칙 (LOBO, 16 평가점)")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)

    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows = []
    for val in TRAIN_NAMES:
        ref = train[train.bearing != val].reset_index(drop=True)
        vsub = train[train.bearing == val].sort_values("t_s").reset_index(drop=True)
        n = len(vsub)
        for frac in [0.25, 0.50, 0.75, 0.90]:
            idx = min(int(n * frac), n - 1)
            qr = vsub.iloc[idx]
            true_rul = float(qr["rul_s"])
            hi = float(qr["HI"]); band = get_band(hi)
            ruls, w = knn_dist(ref, qr)
            c = candidates(ruls, w)
            grid = HI_BAND_GRIDS[band]

            # band_sens (현 1순위 규칙)
            cand_sens = {k: float(np.mean([asym_score([v], [g]) for g in grid]))
                         for k, v in c.items()}
            band_sens_pick = c[max(cand_sens, key=cand_sens.get)]

            # grid_asymopt: KNN 분포 + band grid (동일 가중) 합쳐 asym-opt
            comb_ruls = np.concatenate([ruls, np.array(grid, float)])
            comb_w = np.concatenate([w / w.sum(), np.ones(len(grid)) / len(grid)])
            grid_asymopt = asym_opt(comb_ruls, comb_w)

            rec = {
                "val": val, "frac": frac, "true_rul": true_rul, "band": band,
                "s_asym": asym_score([c["asym"]], [true_rul]),
                "s_q35": asym_score([c["q35"]], [true_rul]),
                "s_median": asym_score([c["median"]], [true_rul]),
                "s_q25": asym_score([c["q25"]], [true_rul]),
                "s_blend_asym_q35": asym_score([0.5*c["asym"]+0.5*c["q35"]], [true_rul]),
                "s_band_sens": asym_score([band_sens_pick], [true_rul]),
                "s_grid_asymopt": asym_score([grid_asymopt], [true_rul]),
            }
            # shrink: asym 을 band median 쪽으로 수축
            band_med = float(np.median(grid))
            for lam in lambdas:
                shr = (1 - lam) * c["asym"] + lam * band_med
                rec[f"s_shrink_{lam}"] = asym_score([shr], [true_rul])
            rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "26_selection_rule_search.csv", index=False)

    score_cols = [c for c in res.columns if c.startswith("s_")]
    means = res[score_cols].mean().sort_values(ascending=False)
    print("\n  규칙별 mean asym (LOBO 16점, 높을수록 좋음):")
    for k, v in means.items():
        marker = "  ← 현 1순위" if k == "s_band_sens" else ""
        print(f"    {k:22s}: {v:.4f}{marker}")

    base = res["s_band_sens"].mean()
    best_rule = means.index[0]
    best_val = means.iloc[0]
    print(f"\n  현 band_sens = {base:.4f}")
    print(f"  최고 규칙   = {best_rule} ({best_val:.4f}, Δ{best_val-base:+.4f})")
    if best_val - base >= 0.02 and best_rule != "s_band_sens":
        print(f"  → ★ {best_rule} 가 +0.02 이상 우월. 1순위 규칙 교체 검토 가치 있음.")
    else:
        print(f"  → 현 band_sens 가 최선권 (개선 <0.02). 변경 불필요 확정.")

    # per-band 최고 규칙
    print("\n  Band별 최고 규칙:")
    for band in ["low", "midlow", "midhigh", "high"]:
        sub = res[res.band == band]
        if len(sub):
            bm = sub[score_cols].mean().sort_values(ascending=False)
            print(f"    {band:8s} (n={len(sub)}): {bm.index[0]}={bm.iloc[0]:.3f} "
                  f"(band_sens={sub['s_band_sens'].mean():.3f})")

    print(f"\n  Saved: {RESULT_DIR / '26_selection_rule_search.csv'}")


if __name__ == "__main__":
    main()
