"""37_PStarEstimator — 비대칭점수-최적 점추정 (p*) flagship 엔진.

핵심: 각 베어링의 마지막 HI에서 train 전체의 HI-KNN(K=20) 이웃의 실제 rul_s 분포를 잡고,
그 분포에 대해 p* = argmax_p E[asym_score(p, R)] 를 푼다.
→ 평가식(asym, 늦은예측 2.5x) 자체를 목적함수로 한 의사결정이론 점추정.
   한 규칙에서 6 베어링 값이 전부 나온다. train-only · 임의 clamp 無 · 600s 물리 하한만 · β 곱셈보정만.

※ 이 스크립트가 발표 헤드라인("asym 직접최적화 argmax_p E[A]")을 *실제로* 구현하는 엔진이다.
   (구 18_PerBearing_Robust = per_bearing_robust.py 는 9개 사전계산 벡터의 메타-셀렉터로, 헤드라인과 절차가 달랐음.)

산출:
  artifacts/results/17_AsymOptimal_TrainBased/37_pstar_submission.xlsx      (flagship 6 베어링)
  artifacts/results/17_AsymOptimal_TrainBased/37_pstar_conservative.xlsx    (p* × β0.97, 600s 하한)
  artifacts/results/17_AsymOptimal_TrainBased/37_pstar_debug.csv            (이웃통계 + p* + E*)
  artifacts/results/17_AsymOptimal_TrainBased/37_pstar_lobo.csv             (LOBO 재현표: fold별 정직 공개)
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
# shared 패키지는 repo에선 ENSEMBLE.parent/shared, code.zip에선 ENSEMBLE/shared 에 있음 → 둘 다 path에.
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
FLOOR = 600.0          # 측정 간격 = 물리 하한 (유일하게 허용된 clip)
BETA = 0.97            # 보수 변형 곱셈 보정 (23_beta_sweep: robust_mean 0.4898 지배)
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]   # LOBO progression 평가점 (600s 끝점 편향 회피)


def knn_neighbors(hi: float, pool: pd.DataFrame, k: int = K) -> np.ndarray:
    """pool 안에서 |HI - hi| 최근접 k개의 실제 rul_s 분포."""
    d = (pool.HI - hi).abs().values
    idx = np.argsort(d)[:k]
    return pool.iloc[idx].rul_s.values


def p_star(hi: float, pool: pd.DataFrame, k: int = K) -> tuple[float, float, np.ndarray]:
    """argmax_p E_neigh[asym(p, r)] — 비대칭점수 기대값 최대 점추정 + 그 기대점수 E*."""
    neigh = knn_neighbors(hi, pool, k)
    grid = np.linspace(max(float(neigh.min()), FLOOR), float(neigh.max()), GRID_N)
    e = [float(np.mean([asym_score(p, r) for r in neigh])) for p in grid]
    j = int(np.argmax(e))
    return float(grid[j]), float(e[j]), neigh


def main() -> None:
    print("=" * 78)
    print("37_PStarEstimator — 비대칭점수-최적 점추정 p* = argmax_p E[asym(p,R)] (flagship)")
    print("=" * 78)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s"]].dropna()

    # ---------- (1) flagship 6 베어링 예측 ----------
    rows, sub_rows, cons_rows = [], [], []
    for b in ORDER:
        hi = float(df[df.bearing == b].HI.iloc[-1])
        ps, es, neigh = p_star(hi, train)
        cons = max(ps * BETA, FLOOR)
        rows.append(dict(
            Bearing=b, HI_last=round(hi, 4),
            neigh_med=round(float(np.median(neigh))), neigh_lo=round(float(neigh.min())),
            neigh_hi=round(float(neigh.max())),
            p_star=round(ps), E_star=round(es, 4), p_star_beta097=round(cons),
        ))
        sub_rows.append(dict(Bearing=b, RUL_pred_seconds=ps, RUL_pred_hours=ps / 3600.0))
        cons_rows.append(dict(Bearing=b, RUL_pred_seconds=cons, RUL_pred_hours=cons / 3600.0))
        print(f"  {b}: HI={hi:.3f}  이웃 med={np.median(neigh):.0f} [{neigh.min():.0f}~{neigh.max():.0f}]"
              f"  → p*={ps:.0f}  (E*={es:.4f})  | β0.97={cons:.0f}")

    debug = pd.DataFrame(rows)
    pd.DataFrame(sub_rows).to_excel(RDIR / "37_pstar_submission.xlsx", index=False)
    pd.DataFrame(cons_rows).to_excel(RDIR / "37_pstar_conservative.xlsx", index=False)
    debug.to_csv(RDIR / "37_pstar_debug.csv", index=False)

    # support 검증 (전부 이웃 [lo,hi] 내 + ≥600 → 임의값/외삽 아님)
    in_support = all(r["neigh_lo"] <= r["p_star"] <= r["neigh_hi"] and r["p_star"] >= FLOOR for r in rows)
    print(f"\n  [support 검증] 모든 p* ∈ [neigh_lo, neigh_hi] 이고 ≥600s: {in_support}")

    # ---------- (2) LOBO 재현표 (정직한 일반화 검증) ----------
    print("\n" + "-" * 78)
    print("  LOBO (Leave-One-Bearing-Out) — held-out 베어링 progression 에 p* 규칙 적용")
    print("  (이웃은 나머지 3 train 베어링에서만 → out-of-bearing 일반화 측정)")
    lobo_rows = []
    fold_means = {}
    for held in TRAINS:
        others = train[train.bearing != held]
        s = train[train.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        scs = []
        for f in FRACS:
            i = int(f * (len(s) - 1))
            hi_pt = float(s.HI.iloc[i]); true = float(s.rul_s.iloc[i])
            ps, _, _ = p_star(hi_pt, others)
            sc = asym_score(ps, true)
            scs.append(sc)
            lobo_rows.append(dict(held=held, frac=f, HI=round(hi_pt, 3),
                                  true_rul=round(true), p_star=round(ps), asym=round(sc, 4)))
        fold_means[held] = float(np.mean(scs))
        print(f"    {held}: fold mean asym = {fold_means[held]:.4f}   (점별: "
              + ", ".join(f"{x:.2f}" for x in scs) + ")")
    overall = float(np.mean(list(fold_means.values())))
    pd.DataFrame(lobo_rows).to_csv(RDIR / "37_pstar_lobo.csv", index=False)
    print(f"\n  LOBO 전체 평균 asym = {overall:.4f}")
    print(f"  fold 분산(정직 공개): min={min(fold_means.values()):.4f} max={max(fold_means.values()):.4f}"
          f"  range={max(fold_means.values()) - min(fold_means.values()):.4f}")
    print("  ※ HI-transfer 가정의 fold별 불균질을 숨기지 않고 노출 — mid-life long은")
    print("    '검증된 사실'이 아니라 '예비 리더보드로 판정할 베팅'으로 프레임할 것.")

    print(f"\n  Saved: 37_pstar_submission.xlsx / 37_pstar_conservative.xlsx / 37_pstar_debug.csv / 37_pstar_lobo.csv")


if __name__ == "__main__":
    main()
