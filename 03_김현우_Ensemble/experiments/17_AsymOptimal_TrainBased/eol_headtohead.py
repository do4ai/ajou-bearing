"""29_EOL_HeadToHead — 두 트랙을 '동일한 held-out EOL 지점'에서 공정 비교.

배경(iter41 정합): 두 트랙의 LOBO가 서로 다른 지점을 평가해 직접 비교 불가였다.
  - A(26_FinalRobust): held-out 베어링의 '실제 EOL(true=600s)'만 평가 → anchor가 6000~16800 예측 → last_score≈0 (4/4 fold).
  - B(per-bearing): frac 0.25~0.9 progression 16점만 평가(EOL 끝점 회피) → mean 0.519.

이 스크립트는 B의 선택 철학(HI-state KNN → argmax_p E[A(p,r)])을 A가 평가한 것과
'동일한 EOL 끝점'에서 돌려서, near-EOL 일반화를 공정 비교한다. train-only, 누수 없음.
(Test5/Test6는 near-EOL 베팅이므로 이 비교가 트랙 우선순위 결정의 핵심 근거.)

출력: artifacts/results/17_AsymOptimal_TrainBased/29_eol_headtohead.csv
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
A_LOBO = ENSEMBLE / "artifacts/results/26_FinalRobust_LOBOFrozenSelector/26_final_robust_lobo.csv"
OUT = ENSEMBLE / "artifacts/results/17_AsymOptimal_TrainBased/29_eol_headtohead.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
K = 20


def asym_optimal_point(neigh_rul: np.ndarray) -> float:
    """argmax_p mean_r A(p, r): 이웃 RUL 분포에서 비대칭 기대점수 최대 단일점."""
    lo, hi = float(neigh_rul.min()), float(neigh_rul.max())
    grid = np.linspace(max(lo, 600.0), hi, 400)
    best_p, best_s = grid[0], -1.0
    for p in grid:
        s = float(np.mean([asym_score(p, r) for r in neigh_rul]))
        if s > best_s:
            best_s, best_p = s, p
    return best_p


def main() -> None:
    print("=" * 72)
    print("29_EOL_HeadToHead — A(frozen anchor) vs B(per-bearing select) @ 동일 EOL 끝점")
    print("=" * 72)

    df = pd.read_csv(FEAT)
    df = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s"]].dropna()
    a_lobo = pd.read_csv(A_LOBO).set_index("bearing")

    rows = []
    for held in TRAINS:
        sub = df[df.bearing == held].sort_values("rul_s")
        last = sub.iloc[0]                      # 최소 rul_s = EOL 끝점
        true_rul = float(last.rul_s)
        hi_q = float(last.HI)

        pool = df[df.bearing != held]           # 나머지 3 베어링 = LOBO train
        d = (pool.HI - hi_q).abs().values
        nn = pool.iloc[np.argsort(d)[:K]]
        neigh = nn.rul_s.values

        # B: 비대칭 최적점 + 참고용 분위수
        p_asymopt = asym_optimal_point(neigh)
        p_q25 = float(np.percentile(neigh, 25))
        p_med = float(np.percentile(neigh, 50))

        sB = asym_score(p_asymopt, true_rul)
        sB_q25 = asym_score(p_q25, true_rul)
        sB_med = asym_score(p_med, true_rul)

        # A: 이미 산출된 frozen-anchor+gate 결과
        sA = float(a_lobo.loc[held, "last_score"])
        predA = float(a_lobo.loc[held, "pred_final_safe"])

        rows.append(dict(
            held=held, true_eol=round(true_rul), HI=round(hi_q, 3),
            A_pred=round(predA), A_score=round(sA, 4),
            B_asymopt_pred=round(p_asymopt), B_asymopt_score=round(sB, 4),
            B_q25_pred=round(p_q25), B_q25_score=round(sB_q25, 4),
            B_med_pred=round(p_med), B_med_score=round(sB_med, 4),
        ))
        print(f"\n  {held}: true_EOL={true_rul:.0f}s  HI={hi_q:.3f}  (이웃 RUL {neigh.min():.0f}~{neigh.max():.0f})")
        print(f"    A frozen-anchor : pred={predA:7.0f}  score={sA:.4f}")
        print(f"    B asym-optimal  : pred={p_asymopt:7.0f}  score={sB:.4f}")
        print(f"    B q25 / median  : {p_q25:7.0f}/{p_med:.0f}  score={sB_q25:.4f}/{sB_med:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n" + "-" * 72)
    print(f"  EOL 끝점 평균 asym_score (4 fold):")
    print(f"    A frozen-anchor : {out.A_score.mean():.4f}")
    print(f"    B asym-optimal  : {out.B_asymopt_score.mean():.4f}")
    print(f"    B q25           : {out.B_q25_score.mean():.4f}")
    print(f"    B median        : {out.B_med_score.mean():.4f}")
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
