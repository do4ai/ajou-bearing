"""31_HI_Transfer_Diagnostic — "HI가 베어링 간 transfer 하는가?"를 train-only로 정량화.

배경(iter40~43): 두 제출 트랙(B=public-optimal, A=anti-overfit)은 정반대 prior에 베팅.
  - A: HI가 transfer 한다(HI0.46→~30000s, train과 동일) → mid-life 길게.
  - B: test는 HI가 시사하는 것보다 빨리 죽는다 → mid-life 짧게.
트랙 선택의 crux = HI→RUL 매핑이 베어링 간 일반화하는가. 이걸 직접 측정한 적은 없어
(iter43은 per-point E[A] 비교였음) 여기서 정식 진단한다. 예측 불변·진단 전용.

측정 (train Train1~4):
  (1) 동일-HI scatter: mid-life band(HI 0.3~0.6)에서 matched-HI RUL의 변동계수(CV).
      CV 크면 = 같은 HI라도 베어링마다 RUL 천차만별 = HI 약한 예측자 = 두 트랙 모두 불확실.
  (2) LOBO HI→RUL 전이력: 3 베어링으로 log(rul)~HI 단조/선형 fit → 4번째 held-out 베어링
      RUL을 HI만으로 예측 → 4-fold MAPE & asym. "HI만 알 때 미지 베어링 RUL 예측 가능한가".

출력: artifacts/results/17_AsymOptimal_TrainBased/31_hi_transfer_diagnostic.csv
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
OUT = ENSEMBLE / "artifacts/results/17_AsymOptimal_TrainBased/31_hi_transfer_diagnostic.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]


def main() -> None:
    print("=" * 76)
    print("31_HI_Transfer_Diagnostic — HI→RUL 베어링 간 전이력 정량화 (train-only)")
    print("=" * 76)

    df = pd.read_csv(FEAT)
    tr = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s"]].dropna()

    # (1) 동일-HI scatter: mid-life band에서 matched HI의 RUL 변동
    print("\n[1] 동일-HI RUL scatter (mid-life band, Test1/2/4/6 영역)")
    print(f"  {'HI 구간':12s} {'n':>4} {'RUL median':>11} {'RUL min~max':>16} {'CV':>6}")
    band_rows = []
    for lo, hi in [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60)]:
        sub = tr[(tr.HI >= lo) & (tr.HI < hi)]
        if len(sub) < 3:
            continue
        r = sub.rul_s.values
        cv = float(r.std() / r.mean())
        n_bear = sub.bearing.nunique()
        print(f"  [{lo:.2f},{hi:.2f})  {len(sub):>4} {np.median(r):>11.0f} "
              f"{r.min():>7.0f}~{r.max():<8.0f} {cv:>6.2f}  ({n_bear}개 베어링)")
        band_rows.append(dict(kind="scatter", band=f"{lo:.2f}-{hi:.2f}", n=len(sub),
                              rul_med=round(float(np.median(r))), rul_min=round(float(r.min())),
                              rul_max=round(float(r.max())), cv=round(cv, 3)))

    # (2) LOBO HI→RUL 전이력: 3베어링 fit → 4번째 예측 (HI만 사용)
    print("\n[2] LOBO HI→RUL 전이 (HI만으로 held-out 베어링 RUL 예측)")
    print("  log(rul) ~ HI 선형 fit (3 베어링) → held-out 베어링 전 측정점 예측")
    print(f"  {'held-out':10s} {'n':>4} {'MAPE(%)':>8} {'asym':>6} {'R²(log)':>8}")
    lobo_rows, mapes, asyms, r2s = [], [], [], []
    for held in TRAINS:
        tr_fit = tr[tr.bearing != held]
        tr_ev = tr[tr.bearing == held]
        x_fit, y_fit = tr_fit.HI.values, np.log(np.clip(tr_fit.rul_s.values, 600, None))
        # 선형 fit log(rul) = a + b*HI
        b, a = np.polyfit(x_fit, y_fit, 1)
        y_ev_true = tr_ev.rul_s.values
        y_ev_pred = np.exp(a + b * tr_ev.HI.values)
        y_ev_pred = np.clip(y_ev_pred, 600, None)
        mape = float(np.mean(np.abs(y_ev_true - y_ev_pred) / np.clip(y_ev_true, 1, None)) * 100)
        asym = asym_score(y_ev_pred, y_ev_true)
        # R² on log scale (held-out)
        yl = np.log(np.clip(y_ev_true, 600, None))
        ss_res = np.sum((yl - (a + b * tr_ev.HI.values)) ** 2)
        ss_tot = np.sum((yl - yl.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        print(f"  {held:10s} {len(tr_ev):>4} {mape:>8.1f} {asym:>6.3f} {r2:>8.3f}")
        mapes.append(mape); asyms.append(asym); r2s.append(r2)
        lobo_rows.append(dict(kind="lobo_transfer", held=held, n=len(tr_ev),
                              mape_pct=round(mape, 1), asym=round(asym, 3), r2_log=round(r2, 3)))

    print("\n" + "-" * 76)
    print(f"  [전이력 요약] 4-fold 평균: MAPE={np.mean(mapes):.0f}%  asym={np.mean(asyms):.3f}  "
          f"R²(log)={np.nanmean(r2s):.2f}")
    print(f"  해석:")
    print(f"   - 동일-HI CV가 크면(>0.3) HI는 약한 RUL 예측자 → A의 'HI transfer' 베팅 위험,")
    print(f"     B의 보수 베팅도 정당. 즉 4 베어링 한계의 irreducible 불확실성.")
    print(f"   - LOBO R²(log)가 낮거나 음수면 HI 단독으론 미지 베어링 RUL 일반화 불가 → 헤지 정합.")

    pd.DataFrame(band_rows + lobo_rows).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
