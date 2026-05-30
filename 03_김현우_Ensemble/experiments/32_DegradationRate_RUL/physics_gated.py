"""35_PhysicsGated — 물리 열화 RUL + 2축 severity gate를 단일 일관 규칙으로 결합 (train-based).

iter46(avg-rate 0.586) + iter48(2축 severity) 종합. **per-bearing test-튜닝이 아니라 단일 규칙**:
  regime(severe) = (energy_ratio > τ_e) OR (rms_multi > τ_r) OR (HI > 0.85)      ← τ = train near-EOL p90
  if severe:  RUL = p_eol*   (train-severe rul_s 의 asym-최적 단일점; train-based, clamp無)
  else:       RUL = elapsed × (1 − HI)/HI   (avg-rate 물리 모델)

전부 train에서 도출된 임계/대표값. LOBO로 **전체 규칙**을 인과 평가 → 0.586 넘는지 + Test 6예측 산출.
출력: artifacts/results/32_DegradationRate_RUL/35_physics_gated.csv
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
OUT = ENSEMBLE / "artifacts/results/32_DegradationRate_RUL/35_physics_gated.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
TESTS = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
STEP_S = 600.0
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]


def hi_at(hi_raw, i, w=3):
    lo = max(0, i - w + 1)
    return float(np.clip(np.median(hi_raw[lo:i + 1]), 1e-3, 0.999))


def asym_opt_point(rul_vals):
    """train-severe rul 분포에서 비대칭 기대점수 최대 단일점 (train-based)."""
    rul_vals = np.asarray(rul_vals, float)
    grid = np.linspace(max(rul_vals.min(), 600), np.percentile(rul_vals, 95), 300)
    best, bs = grid[0], -1
    for p in grid:
        s = float(np.mean([asym_score(p, r) for r in rul_vals]))
        if s > bs:
            bs, best = s, p
    return best


def fit_gate_params(train_df):
    """train near-EOL(rul<=3000)서 severity 임계 + severe 대표 RUL 산출."""
    eol = train_df[train_df.rul_s <= 3000]
    te = float(eol.energy_ratio.quantile(0.90))
    tr = float(eol.rms_multi.quantile(0.90))
    severe_pts = train_df[(train_df.energy_ratio > te) | (train_df.rms_multi > tr) | (train_df.HI > 0.85)]
    p_eol = asym_opt_point(severe_pts.rul_s.values)
    return te, tr, p_eol


def predict(row_hi, elapsed, energy, rms, te, tr, p_eol):
    severe = (energy > te) or (rms > tr) or (row_hi > 0.85)
    if severe:
        return max(p_eol, 600.0), True
    return max(elapsed * (1.0 - row_hi) / row_hi, 600.0), False


def main():
    print("=" * 76)
    print("35_PhysicsGated — 물리 열화 + 2축 severity gate 단일 규칙 (train-based)")
    print("=" * 76)
    df = pd.read_csv(FEAT)

    te0, tr0, peol0 = fit_gate_params(df[df.bearing.isin(TRAINS)])
    print(f"\n  전체 train: severity 임계 energy>{te0:.2f}/rms>{tr0:.3f}, severe 대표 RUL p_eol*={peol0:.0f}s")

    # ---- LOBO: 3개로 gate params 캘리브 → held-out 인과 평가 ----
    print("\n[1] LOBO (3개 캘리브 → held-out progression 인과 평가):")
    rows, lobo = [], []
    for held in TRAINS:
        others = df[df.bearing.isin([b for b in TRAINS if b != held])]
        te, tr, peol = fit_gate_params(others)
        s = df[df.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        scs = []
        for f in FRACS:
            i = int(f * (len(s) - 1))
            hi = hi_at(s.HI.values, i)
            est, sev = predict(hi, i * STEP_S, float(s.energy_ratio.iloc[i]), float(s.rms_multi.iloc[i]), te, tr, peol)
            scs.append(asym_score(est, float(s.rul_s.iloc[i])))
        m = float(np.mean(scs))
        lobo.append(m)
        print(f"  held-out {held}: p_eol*={peol:.0f} → asym={m:.3f}")
    lobo_mean = float(np.mean(lobo))
    print(f"  → LOBO 평균 asym = {lobo_mean:.3f}  (avg-rate-only 0.586 / B 0.519 / A 0.508 대비)")

    # ---- Test 6예측 ----
    print("\n[2] Test1~6 physics-gated 예측:")
    PRED_B = {"Test1": 10067, "Test2": 10998, "Test3": 48900, "Test4": 9545, "Test5": 644, "Test6": 10275}
    PRED_A = {"Test1": 32035, "Test2": 33556, "Test3": 6449, "Test4": 14113, "Test5": 644, "Test6": 3000}
    print(f"  {'Bearing':8s} {'HI':>6} {'severe':>7} {'RUL':>9}   (B / A)")
    for b in TESTS:
        s = df[df.bearing == b].reset_index(drop=True)
        i = len(s) - 1
        hi = hi_at(s.HI.values, i)
        est, sev = predict(hi, i * STEP_S, float(s.energy_ratio.iloc[i]), float(s.rms_multi.iloc[i]), te0, tr0, peol0)
        rows.append(dict(bearing=b, HI=round(hi, 3), severe=bool(sev), rul_gated=round(est),
                         predB=PRED_B[b], predA=PRED_A[b]))
        print(f"  {b:8s} {hi:>6.3f} {str(sev):>7} {est:>9.0f}   (B={PRED_B[b]} / A={PRED_A[b]})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print(f"\n  [판정] LOBO {lobo_mean:.3f} ≥ 0.586 이면 단일 일관 규칙으로 최고 성능 → 신규 후보 자격(사용자 결정).")


if __name__ == "__main__":
    main()
