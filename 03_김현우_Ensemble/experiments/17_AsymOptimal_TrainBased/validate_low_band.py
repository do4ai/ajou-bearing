"""25_ValidateLowBand — Test3 regime (HI<0.30, low band) 명시적 LOBO 검증.

iter18 검증(25~90% progression)은 HI≥0.30만 평가 → low band(Test3) 미검증.
Train 모두 HI<0.30 초기 측정 보유 (RUL 42000~82200s).
이 구간에서 per-bearing 선택 로직이 실제 RUL을 잘 맞히는지 검증.

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/25_low_band_validation.csv
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
MATCH = ["HI", "HI_slope5", "HI_d5", "HI_roll_std5", "rms_multi", "energy_ratio",
         "chsym_max_env_kurt", "chsym_max_kurt"]
K = 20
LOW_GRID = [30000, 50000, 70000, 90000]


def asym_opt(ruls, w, lo=600.0):
    ruls = np.asarray(ruls, float); w = np.asarray(w, float); w = w / (w.sum() + 1e-12)
    hi = float(ruls.max() * 1.2 + 1000)
    def neg(p):
        er = 100.0 * (ruls - p) / (ruls + 1e-12); lh = np.log(0.5)
        a = np.where(er <= 0, np.exp(np.clip(-lh*er/20, -50, 0)), np.exp(np.clip(lh*er/50, -50, 0)))
        return -float(np.sum(w * a))
    return float(minimize_scalar(neg, bounds=(lo, hi), method="bounded").x)


def main() -> None:
    print("=" * 70)
    print("25_ValidateLowBand — Test3 regime (HI<0.30) 검증")
    print("=" * 70)
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)

    rows = []
    for val in TRAIN_NAMES:
        ref = train[train.bearing != val].reset_index(drop=True)
        vsub = train[train.bearing == val].sort_values("t_s").reset_index(drop=True)
        low = vsub[vsub.HI < 0.30]
        if len(low) == 0:
            continue
        # low 구간에서 2~3개 시점 (시작, 중간, HI 0.30 근처)
        picks = [low.index[0], low.index[len(low)//2], low.index[-1]]
        sc = StandardScaler().fit(ref[MATCH].fillna(0).values)
        x_ref = sc.transform(ref[MATCH].fillna(0).values)
        for pi in picks:
            qr = vsub.loc[pi]
            true_rul = float(qr["rul_s"])
            x_q = sc.transform(qr[MATCH].values.reshape(1, -1).astype(float))
            d = pairwise_distances(x_q, x_ref)[0]
            order = np.argsort(d)[:K]
            ruls = ref.iloc[order]["rul_s"].values.astype(float)
            w = 1.0 / (d[order] + 1e-6)
            cands = {"knn_asym": asym_opt(ruls, w),
                     "knn_q35": float(np.percentile(ruls, 35)),
                     "knn_median": float(np.median(ruls)),
                     "knn_q25": float(np.percentile(ruls, 25))}
            cand_sens = {k: float(np.mean([asym_score([v], [g]) for g in LOW_GRID]))
                         for k, v in cands.items()}
            best = max(cand_sens, key=cand_sens.get)
            sel = cands[best]
            rows.append({"val": val, "HI": float(qr["HI"]), "true_rul": true_rul,
                         "selected": best, "sel_pred": sel,
                         "sel_score": asym_score([sel], [true_rul]),
                         "median_score": asym_score([cands["knn_median"]], [true_rul]),
                         "median_pred": cands["knn_median"]})
            print(f"  {val} HI={qr['HI']:.3f}: true={true_rul:.0f}  sel={sel:.0f}({best}) "
                  f"score={rows[-1]['sel_score']:.3f}  median={cands['knn_median']:.0f} "
                  f"score={rows[-1]['median_score']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "25_low_band_validation.csv", index=False)
    print(f"\n  Low-band selection mean asym: {res['sel_score'].mean():.4f}")
    print(f"  Low-band median  mean asym: {res['median_score'].mean():.4f}")
    print(f"\n  [Test3 함의] Test3(HI=0.165) 예측은 low-band selection 신뢰도 = {res['sel_score'].mean():.3f}")
    print(f"  Saved: {RESULT_DIR / '25_low_band_validation.csv'}")


if __name__ == "__main__":
    main()
