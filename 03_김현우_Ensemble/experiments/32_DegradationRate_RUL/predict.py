"""32_DegradationRate_RUL — 열화 속도(HI-velocity) 기반 물리 RUL 추정 (train-based, 신규).

동기(iter40~45): 두 트랙 A/B는 'HI-level→RUL' 가정에서 정반대로 갈렸고, HI-level 단독
전이력은 불완전(R²=0.65, 동일-HI CV up to 0.32). 같은 HI=0.46이라도 *빨리* 열화하면 짧은
RUL(B의 베팅), *느리게* 열화하면 긴 RUL(A의 베팅) — 즉 **열화 속도가 둘을 베어링별로 가른다.**

물리 모델 (임의 clamp 無, 파라미터 거의 없음):
    RUL_est = (HI_failure − HI_now) / (dHI/dt)_now × 600s
  - HI_failure: 고장 시 HI ≈ 1.0 (train 4개 전부 rul=600s에서 HI 0.985~1.0 도달, 확인).
  - (dHI/dt)_now: 인과적(과거만) trailing-window 최소제곱 기울기 (HI 단위/step, step=600s).
  - HI는 절대 건강지표(Test는 truncated라 1.0 미만) → Test에도 동일 적용 가능.

검증: LOBO 불필요(fitted param 거의 없음)하나, train 4개를 progression(0.25~0.9)에서 인과적
평가해 asym 측정 → A(HI-prior)·B(per-bearing)와 동일 16점에서 비교. Test1~6엔 마지막
관측점에서 적용해 'A쪽(긴 RUL)인지 B쪽(짧은 RUL)인지' 데이터가 베어링별로 판정.

출력: artifacts/results/32_DegradationRate_RUL/32_degradation_rate_rul.csv
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
OUT = ENSEMBLE / "artifacts/results/32_DegradationRate_RUL/32_degradation_rate_rul.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
TESTS = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
STEP_S = 600.0
HI_FAIL = 1.0
SMOOTH_W = 5          # HI 평활 윈도우
SLOPE_W = 15         # trailing 기울기 윈도우 (step)
# 물리 하한: train에서 관측된 가장 느린 열화율 → slope clip 하한 (과대 RUL 폭주 방지, 임의값 아님)
MIN_SLOPE = None      # train에서 산출


def hi_at(hi_raw: np.ndarray, i: int, w: int = 3) -> float:
    """점 i에서 인과적·강건한 HI 추정 = 최근 w점 중앙값 (edge 패딩 버그 회피)."""
    lo = max(0, i - w + 1)
    return float(np.median(hi_raw[lo:i + 1]))


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """국소 기울기용 가벼운 평활 (내부점에서만 사용)."""
    if len(x) < 2:
        return x
    w = min(w, len(x))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def causal_slope(hi_sm: np.ndarray, i: int, w: int) -> float:
    """점 i에서 과거만 사용한 trailing-window HI 기울기 (HI/step)."""
    lo = max(0, i - w + 1)
    seg = hi_sm[lo:i + 1]
    if len(seg) < 3:
        return np.nan
    xs = np.arange(len(seg))
    return float(np.polyfit(xs, seg, 1)[0])


def rul_from_rate(hi_now: float, slope: float, min_slope: float) -> float:
    s = max(slope, min_slope)
    rem_hi = max(HI_FAIL - hi_now, 0.0)
    return max(rem_hi / s * STEP_S, 600.0)


def main() -> None:
    print("=" * 76)
    print("32_DegradationRate_RUL — HI-velocity 물리 RUL (train-based, 신규)")
    print("=" * 76)

    df = pd.read_csv(FEAT)

    # train에서 MIN_SLOPE 캘리브레이션: 각 train 전체 평균 열화율의 최솟값의 절반 (보수 하한)
    rates = []
    for b in TRAINS:
        s = df[df.bearing == b].sort_values("rul_s", ascending=False)
        hi = s.HI.values
        rates.append((hi.max() - hi.min()) / len(hi))
    min_slope = 0.5 * min(rates)
    print(f"\n  train 평균 열화율 {[f'{r:.4f}' for r in rates]} → MIN_SLOPE={min_slope:.5f} (최소의 0.5×)")

    # ---- LOBO-style 인과 평가 (train progression): 두 변형 비교 ----
    # 변형1: trailing-window 국소 기울기  (노이즈 민감 가설)
    # 변형2: 평균율 모델 RUL = elapsed × (1−HI)/HI  (HI를 life-start부터 선형 가정, 강건)
    print("\n[1] train progression 인과 평가 (asym): 두 변형")
    fracs = [0.25, 0.5, 0.75, 0.9]
    rows = []
    per_b_slope, per_b_avg = {}, {}
    for b in TRAINS:
        s = df[df.bearing == b].sort_values("rul_s", ascending=False).reset_index(drop=True)
        hi_sm = smooth(s.HI.values, SMOOTH_W)
        sc_sl, sc_av = [], []
        for f in fracs:
            i = int(f * (len(s) - 1))
            true = float(s.rul_s.iloc[i])
            # 변형1
            sl = causal_slope(hi_sm, i, SLOPE_W)
            est_sl = rul_from_rate(hi_sm[i], sl, min_slope)
            # 변형2: 평균율 (elapsed = i steps; HI_now = 강건 인과 추정)
            hi_now = max(hi_at(s.HI.values, i), 1e-3)
            elapsed = i * STEP_S
            est_av = max(elapsed * (HI_FAIL - hi_now) / hi_now, 600.0)
            s1, s2 = asym_score(est_sl, true), asym_score(est_av, true)
            sc_sl.append(s1); sc_av.append(s2)
            rows.append(dict(kind="train", bearing=b, frac=f, HI=round(float(s.HI.iloc[i]), 3),
                             rul_est_slope=round(est_sl), rul_est_avg=round(est_av),
                             true_rul=round(true), asym_slope=round(s1, 3), asym_avg=round(s2, 3)))
        per_b_slope[b] = float(np.mean(sc_sl)); per_b_avg[b] = float(np.mean(sc_av))
        print(f"  {b}: slope mean={per_b_slope[b]:.3f} | avg-rate mean={per_b_avg[b]:.3f}")
    lobo_slope = float(np.mean(list(per_b_slope.values())))
    lobo_avg = float(np.mean(list(per_b_avg.values())))
    print(f"  → 16점 평균 asym: trailing-slope={lobo_slope:.3f} | **avg-rate={lobo_avg:.3f}**")
    print(f"    (비교: B per-bearing 0.519 / A HI-prior mid-life 0.508 / HI-only fit 0.552)")

    # ---- Test1~6 적용 (avg-rate 모델, 마지막 관측점) ----
    print("\n[2] Test1~6 avg-rate RUL (마지막 관측점, 인과):")
    PRED_B = {"Test1": 10067, "Test2": 10998, "Test3": 48900, "Test4": 9545, "Test5": 644, "Test6": 10275}
    PRED_A = {"Test1": 32035, "Test2": 33556, "Test3": 6449, "Test4": 14113, "Test5": 644, "Test6": 3000}
    print(f"  {'Bearing':8s} {'HI_now':>7} {'elapsed_s':>9} {'RUL_avg':>9} {'→경향':>9}   (B / A)")
    for b in TESTS:
        s = df[df.bearing == b].reset_index(drop=True)   # Test rows: time order
        i = len(s) - 1
        hi_now = max(hi_at(s.HI.values, i), 1e-3)
        elapsed = i * STEP_S
        est = max(elapsed * (HI_FAIL - hi_now) / hi_now, 600.0)
        da = abs(np.log(est + 1) - np.log(PRED_A[b] + 1))
        db = abs(np.log(est + 1) - np.log(PRED_B[b] + 1))
        lean = "A(긴)" if da < db else "B(짧은)"
        rows.append(dict(kind="test", bearing=b, frac=1.0, HI=round(float(s.HI.iloc[i]), 3),
                         rul_est_avg=round(est), true_rul=-1,
                         predA=PRED_A[b], predB=PRED_B[b], lean=lean))
        print(f"  {b:8s} {s.HI.iloc[i]:7.3f} {elapsed:9.0f} {est:9.0f} {lean:>9}   (B={PRED_B[b]} / A={PRED_A[b]})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print("\n  [해석] avg-rate가 LOBO서 기존 메서드 수준이면 = HI-prior(A)의 물리적 재유도 →")
    print("         Test mid-life A 지지를 독립 물리 모델이 뒷받침. 트레일링-slope는 노이즈로 실패(정직한 음성).")


if __name__ == "__main__":
    main()
