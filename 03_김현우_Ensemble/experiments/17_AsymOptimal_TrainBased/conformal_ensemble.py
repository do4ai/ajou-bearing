"""44_ConformalEnsemble — flagship 앙상블(avg-rate × p*)의 LOBO-잔차 기반 예측구간 (additive).

iter36 conformal은 구 선택방법 잔차였음. flagship이 앙상블로 바뀌었으니 *현재 flagship*의
예측 불확실성을 정직하게 정량화: LOBO held-out progression 점에서 앙상블 규칙의 percentage error
Er=100·(true−pred)/true 경험분포 → split-conformal 유사 구간을 6 테스트 점추정에 부여.

점추정 불변(순수 additive). 발표/합리성/우수성(불확실성 정량화) 보강.
산출: artifacts/results/17_AsymOptimal_TrainBased/44_conformal_ensemble.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from unified_lobo_comparison import m_pstar, m_avgrate, FEAT, RDIR, TRAINS, FRACS, STEP_S, FLOOR  # noqa: E402
from shared.utils import asym_score  # noqa: E402

ORDER = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]


def ens(hi, e, el, pool):
    """앙상블 = 베어링별 avg-rate × p* 기하평균 (0-param)."""
    return float(np.sqrt(max(m_avgrate(hi, e, el, pool), FLOOR) * max(m_pstar(hi, e, el, pool), FLOOR)))


def main() -> None:
    print("=" * 78)
    print("44_ConformalEnsemble — flagship 앙상블 LOBO-잔차 예측구간 (점추정 불변·additive)")
    print("=" * 78)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s", "energy_ratio"]].dropna()

    # (1) LOBO held-out 점에서 앙상블 Er = 100·(true−pred)/true 경험분포
    ers = []
    for held in TRAINS:
        others = train[train.bearing != held]
        s = train[train.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        for f in FRACS:
            i = int(f * (len(s) - 1))
            hi = float(s.HI.iloc[i]); e = float(s.energy_ratio.iloc[i]); el = i * STEP_S
            true = float(s.rul_s.iloc[i]); pred = ens(hi, e, el, others)
            ers.append(100.0 * (true - pred) / true)
    ers = np.array(ers)
    q = {p: float(np.percentile(ers, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
    print(f"\n  앙상블 LOBO Er(%) 경험분포 (n={len(ers)}점): "
          f"median={q[50]:+.1f}  [10%={q[10]:+.1f}, 90%={q[90]:+.1f}]")
    bias = "구조적 보수(under-predict=early)" if q[50] > 3 else ("구조적 over" if q[50] < -3 else "≈무편향")
    print(f"  중앙 Er {q[50]:+.1f}% → {bias}. (Er>0 = pred<true = 이른 예측, 2.5× late 페널티에 정합)")

    # (2) 테스트 앙상블 점추정 + conformal 구간:  true = pred/(1−Er/100)
    flag = pd.read_excel(RDIR / "42_blend_submission.xlsx").set_index("Bearing")["RUL_pred_seconds"].reindex(ORDER)
    rows = []
    print("\n  베어링별 앙상블 점추정 + 80/90% 예측구간 (시간):")
    for b in ORDER:
        p = float(flag[b])
        def at(erq):  # Er 분위 → 함의 true = pred/(1−Er/100)
            return max(p / (1.0 - erq / 100.0), FLOOR)
        # 낮은 Er(=pred 과대)→true 작음=하한 ; 높은 Er(=pred 과소)→true 큼=상한
        lo90, hi90 = at(q[5]), at(q[95])
        lo80, hi80 = at(q[10]), at(q[90])
        rows.append(dict(bearing=b, point_s=round(p), lo80_s=round(lo80), hi80_s=round(hi80),
                         lo90_s=round(lo90), hi90_s=round(hi90),
                         point_h=round(p / 3600, 2), lo90_h=round(lo90 / 3600, 2), hi90_h=round(hi90 / 3600, 2)))
        print(f"    {b}: {p/3600:5.2f}h  80%[{lo80/3600:5.2f}, {hi80/3600:5.2f}]h  "
              f"90%[{lo90/3600:5.2f}, {hi90/3600:5.2f}]h")
    pd.DataFrame(rows).to_csv(RDIR / "44_conformal_ensemble.csv", index=False)
    print("\n  [해석] 구간 폭이 큰 것은 train 4 베어링 한계의 정직한 반영(과대정밀 회피=합리성).")
    print("         점추정은 제출본 불변 — 순수 additive 보강(발표 불확실성 정량화).")
    print(f"\n  Saved: {RDIR / '44_conformal_ensemble.csv'}")


if __name__ == "__main__":
    main()
