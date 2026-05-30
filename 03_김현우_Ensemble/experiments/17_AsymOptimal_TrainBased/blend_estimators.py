"""42_BlendEstimators — 고정(0-param) 블렌드가 단일 최선(avg-rate 0.600)을 이기는가?

동기: 정확도(사용자 #1축)에서 avg-rate(0.600)·p*(0.538)·HI-reg(0.598)가 방향 일치하나 크기 상이.
독립 추정기의 **고정 기하평균(log-space, RUL 스케일 적합)** 은 파라미터 추가 없이 분산을 줄여
단일 최선을 능가할 수 있음(고전적 variance reduction). 단 n=4라 *학습된 가중치*는 과적합 →
**0-param 고정 블렌드만** 시험(학습 weight 금지). 동일 LOBO 점 + 부트스트랩 CI로 공정 판정.

채택 기준: 블렌드 LOBO 평균 > avg-rate(0.600) **이고** P(blend>avg-rate)≥0.6 (노이즈 내 아님).
산출: artifacts/results/17_AsymOptimal_TrainBased/42_blend_lobo.csv
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from unified_lobo_comparison import (  # noqa: E402  (동일 base 추정기·로직 재사용)
    m_pstar, m_avgrate, m_hireg, m_knnq35, FEAT, RDIR, TRAINS, FRACS, STEP_S, FLOOR)
from shared.utils import asym_score  # noqa: E402

BASE = {"p_star": m_pstar, "avg_rate": m_avgrate, "HI_reg": m_hireg, "KNN_q35": m_knnq35}


def geomean(*vs):
    return float(np.exp(np.mean([np.log(max(v, FLOOR)) for v in vs])))


# 고정 블렌드 정의 (전부 0-param)
BLENDS = {
    "avg×pstar (geo)":        lambda d: geomean(d["avg_rate"], d["p_star"]),
    "avg×HIreg (geo)":        lambda d: geomean(d["avg_rate"], d["HI_reg"]),
    "avg×pstar×HIreg (geo)":  lambda d: geomean(d["avg_rate"], d["p_star"], d["HI_reg"]),
    "avg×pstar×q35 (geo)":    lambda d: geomean(d["avg_rate"], d["p_star"], d["KNN_q35"]),
    "all4 (geo)":             lambda d: geomean(*[d[k] for k in BASE]),
}


def main() -> None:
    print("=" * 80)
    print("42_BlendEstimators — 고정(0-param) 블렌드 vs 단일 최선(avg-rate) 공정 LOBO")
    print("=" * 80)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s", "energy_ratio"]].dropna()

    methods = list(BASE) + list(BLENDS)
    fold = {m: {} for m in methods}
    for held in TRAINS:
        others = train[train.bearing != held]
        s = train[train.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        acc = {m: [] for m in methods}
        for f in FRACS:
            i = int(f * (len(s) - 1))
            hi = float(s.HI.iloc[i]); e = float(s.energy_ratio.iloc[i])
            el = i * STEP_S; true = float(s.rul_s.iloc[i])
            base_pred = {k: fn(hi, e, el, others) for k, fn in BASE.items()}
            for k in BASE:
                acc[k].append(asym_score(base_pred[k], true))
            for nm, fn in BLENDS.items():
                acc[nm].append(asym_score(fn(base_pred), true))
        for m in methods:
            fold[m][held] = float(np.mean(acc[m]))

    idx = list(itertools.product(range(4), repeat=4))
    boot = {m: np.array([np.mean([fold[m][TRAINS[j]] for j in s]) for s in idx]) for m in methods}

    rows = []
    print(f"\n  {'method':24} {'mean':>6}  {'95% CI':>16}  P(>avg-rate)")
    for m in sorted(methods, key=lambda x: -np.mean(list(fold[x].values()))):
        mean = float(np.mean(list(fold[m].values())))
        lo, hi = np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)
        pwin = float(np.mean(boot[m] > boot["avg_rate"])) if m != "avg_rate" else float("nan")
        print(f"  {m:24} {mean:6.3f}  [{lo:.3f},{hi:.3f}]   {('-' if m=='avg_rate' else f'{pwin:.2f}')}")
        rows.append(dict(method=m, mean=round(mean, 4), ci_lo=round(float(lo), 4),
                         ci_hi=round(float(hi), 4), p_gt_avgrate=("" if m == "avg_rate" else round(pwin, 2))))
    pd.DataFrame(rows).to_csv(RDIR / "42_blend_lobo.csv", index=False)

    base_avg = float(np.mean(list(fold["avg_rate"].values())))
    winners = [r for r in rows if r["method"] in BLENDS and r["mean"] > base_avg
               and r["p_gt_avgrate"] != "" and r["p_gt_avgrate"] >= 0.6]
    print("\n  [판정]", end=" ")
    if winners:
        w = max(winners, key=lambda r: r["mean"])
        print(f"블렌드 '{w['method']}' (mean {w['mean']}, P>avg={w['p_gt_avgrate']}) > avg-rate {base_avg:.3f}")
        print("           → 0-param 블렌드가 robust 우위 → 정확도 anchor 후보 승격 검토.")
    else:
        print(f"어떤 고정 블렌드도 avg-rate({base_avg:.3f})를 robust(P≥0.6) 능가 못함 → 정직한 음성, avg-rate 유지.")
    print(f"\n  Saved: {RDIR / '42_blend_lobo.csv'}")


if __name__ == "__main__":
    main()
