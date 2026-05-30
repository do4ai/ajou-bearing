"""33_DegradationConvex — 열화 곡률(convexity) 보정 물리 RUL (1-파라미터, LOBO 캘리브).

iter46 발견: 선형-from-origin avg-rate(RUL=t·(1−HI)/HI)는 LOBO 0.586(최고)이나 HI 가속
(EOL 근처 convex)을 무시해 체계적 과대예측. 1-파라미터 곡률 모델로 보정:

    HI(t) = (t/T)^p   (p≥1; p=1이면 선형=avg-rate)
    ⇒ T = t / HI^(1/p),  RUL = T − t = t·(HI^(−1/p) − 1)

p는 **LOBO로 캘리브**(3 베어링서 progression asym 최대화 → 4번째 평가). 1 파라미터·train-based·
임의clamp無. p>1이면 RUL 짧아짐(가속 반영). 목표: avg-rate 0.586을 넘는 robust 추정 + Test 재판정.

출력: artifacts/results/32_DegradationRate_RUL/33_convex_calibrated.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ENSEMBLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENSEMBLE.parent))
from shared.utils import asym_score  # noqa: E402

FEAT = ENSEMBLE / "artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_features_dynamics.csv"
OUT = ENSEMBLE / "artifacts/results/32_DegradationRate_RUL/33_convex_calibrated.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
TESTS = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
STEP_S = 600.0
P_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]  # progression 평가점 (600s 끝점 제외 — 측정하한 degenerate)


def hi_at(hi_raw: np.ndarray, i: int, w: int = 3) -> float:
    lo = max(0, i - w + 1)
    return float(np.clip(np.median(hi_raw[lo:i + 1]), 1e-3, 0.999))


def rul_convex(hi_now: float, elapsed: float, p: float) -> float:
    return max(elapsed * (hi_now ** (-1.0 / p) - 1.0), 600.0)


def eval_p_on(bearings, df, p) -> float:
    """주어진 베어링들의 progression 점에서 평균 asym."""
    scs = []
    for b in bearings:
        s = df[df.bearing == b].sort_values("rul_s", ascending=False).reset_index(drop=True)
        for f in FRACS:
            i = int(f * (len(s) - 1))
            est = rul_convex(hi_at(s.HI.values, i), i * STEP_S, p)
            scs.append(asym_score(est, float(s.rul_s.iloc[i])))
    return float(np.mean(scs))


def main() -> None:
    print("=" * 76)
    print("33_DegradationConvex — 곡률 보정 물리 RUL (1-param, LOBO 캘리브)")
    print("=" * 76)
    df = pd.read_csv(FEAT)

    # ---- in-sample p 곡선 (전체 train) ----
    print("\n[1] 전체 train progression asym vs p:")
    for p in P_GRID:
        print(f"  p={p:<4}: asym={eval_p_on(TRAINS, df, p):.3f}")

    # ---- LOBO 캘리브: held-out마다 나머지 3개로 p* 선택 ----
    print("\n[2] LOBO (3개로 p* 선택 → held-out 평가):")
    rows, lobo_scs = [], []
    for held in TRAINS:
        others = [b for b in TRAINS if b != held]
        p_star = max(P_GRID, key=lambda p: eval_p_on(others, df, p))
        sc = eval_p_on([held], df, p_star)
        lobo_scs.append(sc)
        print(f"  held-out {held}: p*={p_star} (3개 캘리브) → held-out asym={sc:.3f}")
        rows.append(dict(kind="lobo", held=held, p_star=p_star, asym=round(sc, 3)))
    lobo_mean = float(np.mean(lobo_scs))
    print(f"  → LOBO 평균 asym = {lobo_mean:.3f}  (avg-rate p=1 기준 0.586 대비)")

    # 전체 train 최적 p (Test 적용용)
    p_final = max(P_GRID, key=lambda p: eval_p_on(TRAINS, df, p))
    print(f"\n  전체 train 최적 p* = {p_final}")

    # ---- Test 적용 ----
    print(f"\n[3] Test1~6 convex RUL (p={p_final}):")
    PRED_B = {"Test1": 10067, "Test2": 10998, "Test3": 48900, "Test4": 9545, "Test5": 644, "Test6": 10275}
    PRED_A = {"Test1": 32035, "Test2": 33556, "Test3": 6449, "Test4": 14113, "Test5": 644, "Test6": 3000}
    print(f"  {'Bearing':8s} {'HI_now':>7} {'RUL_convex':>11} {'avg-rate(p1)':>13}   (B / A)")
    for b in TESTS:
        s = df[df.bearing == b].reset_index(drop=True)
        i = len(s) - 1
        hi_now = hi_at(s.HI.values, i)
        est = rul_convex(hi_now, i * STEP_S, p_final)
        est_lin = rul_convex(hi_now, i * STEP_S, 1.0)
        rows.append(dict(kind="test", bearing=b, HI=round(hi_now, 3), p_star=p_final,
                         rul_convex=round(est), rul_linear=round(est_lin),
                         predA=PRED_A[b], predB=PRED_B[b]))
        print(f"  {b:8s} {hi_now:7.3f} {est:11.0f} {est_lin:13.0f}   (B={PRED_B[b]} / A={PRED_A[b]})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
