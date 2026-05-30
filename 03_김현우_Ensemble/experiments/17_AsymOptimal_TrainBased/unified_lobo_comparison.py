"""41_UnifiedLOBO — 모든 train-based 점추정 규칙을 *동일* held-out 점에서 공정 비교.

문제: 그동안 후보 LOBO 수치(p* 0.5377 / ① 0.519 / avg-rate 0.586 / ⑥ 0.750…)가 서로 다른
프로토콜·평가점에서 나와 **사과-대-사과 비교가 아니었다**. 특히 사용자 1순위 목표="파이널 스코어
최고점"인데, flagship p*가 정확도상 정말 최선인지(혹은 물리 avg-rate가 더 나은지) 미확정.

본 하니스: 4 train 베어링 LOBO, 동일 FRACS 평가점에서 아래 train-based 점추정기를 전부 채점 +
4-베어링 부트스트랩(256 전수) CI. → p* 채택이 정확도상 방어되는지 정직 판정.
(주의: ①(메타-셀렉터)·③/A(학습 NN)는 임의 progression 점에 재적용 불가[상위모델 필요]라 제외 —
 본 비교는 '임의 점에 적용 가능한 train-분포 점추정 규칙' 군 내 공정 비교.)

점추정기: p*(HI-KNN argmax E[asym]) / p*2축 / avg-rate 물리 / HI회귀 / KNN-median / KNN-q35.
산출: artifacts/results/17_AsymOptimal_TrainBased/41_unified_lobo.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import itertools
from pathlib import Path

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
K = 20
GRID_N = 400
FLOOR = 600.0
STEP_S = 600.0
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]


def _argmax_asym(neigh: np.ndarray) -> float:
    grid = np.linspace(max(float(neigh.min()), FLOOR), float(neigh.max()), GRID_N)
    e = [float(np.mean([asym_score(p, r) for r in neigh])) for p in grid]
    return float(grid[int(np.argmax(e))])


def _knn(hi, pool, k=K):
    d = (pool.HI.values - hi).__abs__()
    return pool.iloc[np.argsort(d)[:k]].rul_s.values


def _knn2(hi, e, pool, k=K):
    zhi = pool.HI.values; le = np.log1p(pool.energy_ratio.values)
    hm, hs = zhi.mean(), zhi.std() or 1.0
    lm, ls = le.mean(), le.std() or 1.0
    d = np.sqrt(((zhi - hm) / hs - (hi - hm) / hs) ** 2 + ((le - lm) / ls - (np.log1p(e) - lm) / ls) ** 2)
    return pool.iloc[np.argsort(d)[:k]].rul_s.values


# 점추정 규칙들: (hi, energy, elapsed, pool) -> pred  (전부 600s 하한)
def m_pstar(hi, e, el, pool):    return _argmax_asym(_knn(hi, pool))
def m_pstar2(hi, e, el, pool):   return _argmax_asym(_knn2(hi, e, pool))
def m_avgrate(hi, e, el, pool):  return max(el * (1 - hi) / hi, FLOOR)
def m_knnmed(hi, e, el, pool):   return max(float(np.median(_knn(hi, pool))), FLOOR)
def m_knnq35(hi, e, el, pool):   return max(float(np.percentile(_knn(hi, pool), 35)), FLOOR)
def m_hireg(hi, e, el, pool):
    b, a = np.polyfit(pool.HI.values, np.log(np.clip(pool.rul_s.values, 600, None)), 1)
    return max(float(np.exp(a + b * hi)), FLOOR)

METHODS = {"p_star": m_pstar, "p_star_2ax": m_pstar2, "avg_rate": m_avgrate,
           "HI_reg": m_hireg, "KNN_med": m_knnmed, "KNN_q35": m_knnq35}


def main() -> None:
    print("=" * 82)
    print("41_UnifiedLOBO — train-based 점추정기 동일-점 공정 비교 + 부트스트랩 CI")
    print("=" * 82)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s", "energy_ratio"]].dropna()

    # fold(베어링)별 method 평균 asym
    fold_score = {m: {} for m in METHODS}
    for held in TRAINS:
        others = train[train.bearing != held]
        s = train[train.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        acc = {m: [] for m in METHODS}
        for f in FRACS:
            i = int(f * (len(s) - 1))
            hi = float(s.HI.iloc[i]); e = float(s.energy_ratio.iloc[i])
            el = i * STEP_S; true = float(s.rul_s.iloc[i])
            for m, fn in METHODS.items():
                acc[m].append(asym_score(fn(hi, e, el, others), true))
        for m in METHODS:
            fold_score[m][held] = float(np.mean(acc[m]))

    # 부트스트랩(4^4=256 전수) CI + 평균
    idx = list(itertools.product(range(4), repeat=4))
    boot = {m: np.array([np.mean([fold_score[m][TRAINS[j]] for j in s]) for s in idx]) for m in METHODS}

    rows = []
    print(f"\n  {'method':12} " + " ".join(f"{b:>7}" for b in TRAINS) + f"  {'mean':>6}  {'95% CI':>16}")
    ranked = sorted(METHODS, key=lambda m: -np.mean(list(fold_score[m].values())))
    for m in ranked:
        v = [fold_score[m][b] for b in TRAINS]
        lo, hi = np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)
        print(f"  {m:12} " + " ".join(f"{x:7.3f}" for x in v)
              + f"  {np.mean(v):6.3f}  [{lo:.3f},{hi:.3f}]")
        rows.append(dict(method=m, **{b: round(fold_score[m][b], 4) for b in TRAINS},
                         mean=round(float(np.mean(v)), 4), ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4)))
    pd.DataFrame(rows).to_csv(RDIR / "41_unified_lobo.csv", index=False)

    # p* vs 최상위 경쟁자 승률
    best_other = next(m for m in ranked if m != "p_star")
    p_ps = float(np.mean(boot["p_star"] > boot[best_other]))
    p_ar = float(np.mean(boot["p_star"] > boot["avg_rate"]))
    print(f"\n  P(p_star > {best_other}) = {p_ps:.2f}   |   P(p_star > avg_rate) = {p_ar:.2f}")

    top = ranked[0]
    print("\n  [판정]", end=" ")
    if top == "p_star":
        print(f"p_star 가 공정 LOBO 평균 1위 → 정확도상 flagship 정당.")
    else:
        gap = np.mean(list(fold_score[top].values())) - np.mean(list(fold_score['p_star'].values()))
        if p_ar < 0.5 or gap > 0.03:
            print(f"⚠ {top}(평균 {np.mean(list(fold_score[top].values())):.3f})가 p_star({np.mean(list(fold_score['p_star'].values())):.3f})보다 LOBO 우위(Δ{gap:.3f}, P(p*>avg)={p_ar:.2f}).")
            print(f"           → flagship p* 는 '발표 seam-free·창의성' 근거로 선택됐으나 정확도 단독 최선은 아닐 수 있음.")
            print(f"           정직 결론: 예비(6/1~5)에서 p* vs {top} 직접 실측 비교 권장. CI 중첩이면 우열 비결정.")
        else:
            print(f"{top}가 근소 우위이나 CI 중첩·Δ{gap:.3f}<0.03 → p_star 와 사실상 동급, flagship 유지 방어 가능.")
    print(f"\n  Saved: {RDIR / '41_unified_lobo.csv'}")


if __name__ == "__main__":
    main()
