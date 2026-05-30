"""30_MidLife_HeadToHead — 두 트랙을 'HI-band prior' 관점에서 공정 비교 (mid-life 노출).

배경: iter42 EOL 공정비교는 '600s 끝점' 관점(B에 유리). 그러나 두 트랙의 최대 충돌은
mid-life 4 베어링(Test1/2/4/6 ~3× 차, Test3 7.6× 역전). 이 관점도 평가해야 균형.

방법(train-only, HI-band prior = "동일 HI면 동일 RUL 분포" 가정):
  각 Test 베어링의 last-HI 에서 train 전체의 HI-KNN(K=20) 이웃의 실제 rul_s 분포를 잡고,
  두 트랙이 '제출한' 예측값에 대해 E[asym_score(pred, r)] (이웃 분포 기대 점수)을 계산.
  ※ 비순환성: B의 mid-life 값은 28_EOLReg 출력, A는 5_HIBlend anchor — 둘 다 이 KNN
    분포의 argmax가 아니므로 어느 쪽도 이 목적함수로 '훈련'되지 않았다 → 공정.

핵심 해석: 이 prior에서 우열이 iter18 600s-LOBO 우열과 반대로 나오면, insight#2의
'LOBO ↔ Sensitivity anti-correlation'이 정량 확인됨 → 트랙 선택 = 어느 prior에 베팅하느냐.

출력: artifacts/results/17_AsymOptimal_TrainBased/30_midlife_headtohead.csv
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
OUT = ENSEMBLE / "artifacts/results/17_AsymOptimal_TrainBased/30_midlife_headtohead.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
ORDER = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
K = 20

# 제출본 값 (검증된 source)
PRED_B = {"Test1": 10067, "Test2": 10998, "Test3": 48900, "Test4": 9545, "Test5": 644, "Test6": 10275}
PRED_A = {"Test1": 32035, "Test2": 33556, "Test3": 6449, "Test4": 14113, "Test5": 644, "Test6": 3000}


def exp_score(pred: float, neigh: np.ndarray) -> float:
    return float(np.mean([asym_score(pred, r) for r in neigh]))


def main() -> None:
    print("=" * 76)
    print("30_MidLife_HeadToHead — HI-band prior 하 기대 asym_score (B 제출 vs A 제출)")
    print("=" * 76)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s"]].dropna()

    rows = []
    for b in ORDER:
        hi = float(df[df.bearing == b].HI.iloc[-1])
        d = (train.HI - hi).abs().values
        neigh = train.iloc[np.argsort(d)[:K]].rul_s.values

        eb = exp_score(PRED_B[b], neigh)
        ea = exp_score(PRED_A[b], neigh)
        # 이 prior 자체의 천장: argmax_p E[A] (어느 트랙도 도달 못한 이론 최적)
        grid = np.linspace(max(neigh.min(), 600), neigh.max(), 400)
        p_star = grid[int(np.argmax([exp_score(p, neigh) for p in grid]))]
        e_star = exp_score(p_star, neigh)

        winner = "B" if eb > ea else ("A" if ea > eb else "tie")
        rows.append(dict(
            bearing=b, HI=round(hi, 3),
            neigh_med=round(float(np.median(neigh))), neigh_lo=round(float(neigh.min())), neigh_hi=round(float(neigh.max())),
            predB=PRED_B[b], E_B=round(eb, 4), predA=PRED_A[b], E_A=round(ea, 4),
            winner=winner, p_star=round(float(p_star)), E_star=round(e_star, 4),
        ))
        print(f"\n  {b}: HI={hi:.3f}  이웃RUL med={np.median(neigh):.0f} [{neigh.min():.0f}~{neigh.max():.0f}]")
        print(f"    B pred={PRED_B[b]:6d}  E[A]={eb:.4f}    A pred={PRED_A[b]:6d}  E[A]={ea:.4f}    → {winner} 우위")
        print(f"    (이 prior 천장 p*={p_star:.0f}, E*={e_star:.4f})")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    mid = out[out.bearing.isin(["Test1", "Test2", "Test4", "Test6"])]   # mid-life
    print("\n" + "-" * 76)
    print(f"  [HI-band prior 관점 기대 asym_score]")
    print(f"    전체 6 평균:    B={out.E_B.mean():.4f}   A={out.E_A.mean():.4f}")
    print(f"    mid-life 4 평균: B={mid.E_B.mean():.4f}   A={mid.E_A.mean():.4f}  (Test1/2/4/6)")
    print(f"    Test3 (저HI):    B={out[out.bearing=='Test3'].E_B.values[0]:.4f}   A={out[out.bearing=='Test3'].E_A.values[0]:.4f}")
    print(f"\n  ★ iter18 600s-LOBO 관점에선 B 우위(0.519>naive). 이 HI-prior 관점 결과와")
    print(f"    비교하면 insight#2(LOBO↔Sensitivity anti-corr) 정량 확인 여부 판단 가능.")
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
