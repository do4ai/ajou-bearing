"""39_PStar2Axis — p* 비대칭점수-최적 점추정의 2축(HI × energy-severity) 조건화 변형.

동기: HI-only p*(37)의 최대 약점 셀 = Test6. HI=0.41(mid)인데 energy_ratio=23.3
(train near-EOL p90=10.8의 2배 = 숨은 급성 열화). HI-only KNN은 HI≈0.41 이웃(=mid-life,
긴 RUL)만 잡아 p*=41400(긺)을 내지만, 2축(HI,energy) KNN은 'mid-HI지만 고energy=near-EOL'
이웃을 잡아 짧은 RUL로 자연 수렴할 것 → 손보정(overfit) 없이 동일 argmax 프레임 안에서 T6 해소 기대.

규칙(동일): p* = argmax_p E_neigh[asym(p,R)]. 차이는 이웃 정의뿐 —
  거리 = z(HI), z(log1p(energy_ratio)) 의 표준화 유클리드 (train 통계로만 fit, test 누수 없음).

★ 채택 기준(anti-overfit 규율): LOBO 전체 asym 이 HI-only(0.5377) **이상**이고, T6가
   원리적으로(손보정 없이) 해소될 때만 flagship 승격 후보. 아니면 정직한 음성 → HI-only 유지.

산출: artifacts/results/17_AsymOptimal_TrainBased/39_pstar_2axis.csv (+ _lobo.csv)
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ENSEMBLE = Path(__file__).resolve().parents[2]
for _p in (ENSEMBLE, ENSEMBLE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from shared.utils import asym_score  # noqa: E402

FEAT = ENSEMBLE / "artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_features_dynamics.csv"
RDIR = ENSEMBLE / "artifacts/results/17_AsymOptimal_TrainBased"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
ORDER = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
K = 20
GRID_N = 400
FLOOR = 600.0
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
COLS = ["bearing", "HI", "rul_s", "energy_ratio"]


def fit_scaler(pool: pd.DataFrame):
    hi = pool.HI.values.astype(float)
    le = np.log1p(pool.energy_ratio.values.astype(float))
    return (hi.mean(), hi.std() or 1.0, le.mean(), le.std() or 1.0)


def neigh_rul(hi: float, e: float, pool: pd.DataFrame, sc, k: int, two_axis: bool) -> np.ndarray:
    zhi = (pool.HI.values - sc[0]) / sc[1]
    q_hi = (hi - sc[0]) / sc[1]
    if two_axis:
        zle = (np.log1p(pool.energy_ratio.values) - sc[2]) / sc[3]
        q_le = (np.log1p(e) - sc[2]) / sc[3]
        d = np.sqrt((zhi - q_hi) ** 2 + (zle - q_le) ** 2)
    else:
        d = np.abs(zhi - q_hi)
    return pool.iloc[np.argsort(d)[:k]].rul_s.values


def p_star(neigh: np.ndarray) -> float:
    grid = np.linspace(max(float(neigh.min()), FLOOR), float(neigh.max()), GRID_N)
    e = [float(np.mean([asym_score(p, r) for r in neigh])) for p in grid]
    return float(grid[int(np.argmax(e))])


def lobo(df_train: pd.DataFrame, two_axis: bool):
    fold = {}
    for held in TRAINS:
        others = df_train[df_train.bearing != held]
        sc = fit_scaler(others)
        s = df_train[df_train.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        scs = []
        for f in FRACS:
            i = int(f * (len(s) - 1))
            ps = p_star(neigh_rul(float(s.HI.iloc[i]), float(s.energy_ratio.iloc[i]), others, sc, K, two_axis))
            scs.append(asym_score(ps, float(s.rul_s.iloc[i])))
        fold[held] = float(np.mean(scs))
    return fold, float(np.mean(list(fold.values())))


def main() -> None:
    print("=" * 80)
    print("39_PStar2Axis — 2축(HI × energy) 조건화 p* vs HI-only p* (정직한 LOBO 비교)")
    print("=" * 80)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][COLS].dropna()
    sc_full = fit_scaler(train)

    # --- 6 테스트 예측: HI-only vs 2-axis ---
    rows = []
    print("\n  베어링별 p* (HI-only → 2-axis):")
    for b in ORDER:
        last = df[df.bearing == b].iloc[-1]
        hi, e = float(last.HI), float(last.energy_ratio)
        n1 = neigh_rul(hi, e, train, sc_full, K, two_axis=False)
        n2 = neigh_rul(hi, e, train, sc_full, K, two_axis=True)
        p1, p2 = p_star(n1), p_star(n2)
        rows.append(dict(bearing=b, HI=round(hi, 3), energy=round(e, 2),
                         hi_only_neigh_med=round(float(np.median(n1))), p_hi_only=round(p1),
                         twoaxis_neigh_med=round(float(np.median(n2))), p_2axis=round(p2)))
        flag = "  ← T6 약점 셀" if b == "Test6" else ""
        print(f"    {b}: HI={hi:.2f} E={e:5.2f} | HI-only p*={p1:6.0f} (med {np.median(n1):.0f})"
              f"  →  2axis p*={p2:6.0f} (med {np.median(n2):.0f}){flag}")
    pd.DataFrame(rows).to_csv(RDIR / "39_pstar_2axis.csv", index=False)

    # --- LOBO 정직 비교 ---
    f1, m1 = lobo(train, two_axis=False)
    f2, m2 = lobo(train, two_axis=True)
    lobo_rows = [dict(method="HI_only", **{k: round(v, 4) for k, v in f1.items()}, overall=round(m1, 4)),
                 dict(method="2axis", **{k: round(v, 4) for k, v in f2.items()}, overall=round(m2, 4))]
    pd.DataFrame(lobo_rows).to_csv(RDIR / "39_pstar_2axis_lobo.csv", index=False)

    print("\n  LOBO (out-of-bearing, 동일 progression 평가점):")
    print(f"    HI-only : " + "  ".join(f"{k}={f1[k]:.3f}" for k in TRAINS) + f"  | 전체={m1:.4f}")
    print(f"    2-axis  : " + "  ".join(f"{k}={f2[k]:.3f}" for k in TRAINS) + f"  | 전체={m2:.4f}")

    # --- 정직한 판정 ---
    t6 = next(r for r in rows if r["bearing"] == "Test6")
    t6_resolved = t6["p_2axis"] < 0.5 * t6["p_hi_only"]   # 절반 이하로 짧아지면 'energy가 끌어내림'
    print("\n  " + "-" * 76)
    print(f"  [판정] LOBO 2axis {m2:.4f} {'≥' if m2 >= m1 else '<'} HI-only {m1:.4f}"
          f"  (Δ={m2 - m1:+.4f}) | T6: {t6['p_hi_only']}→{t6['p_2axis']} "
          f"{'(짧아짐=energy 반영)' if t6_resolved else '(변화 작음)'}")
    if m2 >= m1 - 0.01 and t6_resolved:
        print("  → 2축이 일반화 유지/개선 + T6를 원리적으로 해소 → flagship 2축 변형 승격 검토 가치.")
    else:
        print("  → 채택 기준 미달(LOBO 개선 없음 또는 T6 미해소) → HI-only p* 유지(정직한 음성).")
    print(f"\n  Saved: 39_pstar_2axis.csv / 39_pstar_2axis_lobo.csv")


if __name__ == "__main__":
    main()
