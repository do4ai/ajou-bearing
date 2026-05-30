"""27_ConformalIntervals — LOBO 잔차 기반 conformal 예측구간 (additive, 점추정 불변).

배경: 주류 RUL 문헌은 불확실성 정량화를 강조(iter27 CREATIVITY_POSITIONING).
우리는 의사결정이론적 점추정이 메인이지만, LOBO 잔차로 calibrated 구간을 ALSO 제공해
(a) 정직한 불확실성, (b) 보수 bias(asym 정합) 정량화 를 발표/합리성 근거로 확보.

방법 (split-conformal 유사, train-only):
  - 24_selection_validation.csv 의 16 LOBO 평가점에서 percentage-error
    Er = 100·(Act−Pred)/Act 의 경험분포를 conformity score로 사용.
  - 점추정 pred 에 대한 actual RUL 구간: Act = pred / (1 − Er/100).
  - Er 분위수로 하한/상한 산출. 점추정 자체는 제출본 그대로 (불변).

핵심 통찰: median Er > 0 (under-predict) ⇒ 선택법이 구조적으로 보수(early) ⇒
늦은 예측 2.5× 페널티에 정합. 이는 우연이 아니라 asym-optimal 점추정의 자연 귀결.

출력: artifacts/results/17_AsymOptimal_TrainBased/27_conformal_intervals.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, result_dir  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
LOBO_CSV = RESULT_DIR / "24_selection_validation.csv"
SUB_XLSX = RESULT_DIR / "18_per_bearing_robust_submission.xlsx"


def main() -> None:
    print("=" * 70)
    print("27_ConformalIntervals — LOBO 잔차 기반 예측구간 (점추정 불변)")
    print("=" * 70)

    lobo = pd.read_csv(LOBO_CSV)
    lobo["Er"] = 100.0 * (lobo.true_rul - lobo.sel_pred) / lobo.true_rul
    er = lobo["Er"].values

    # 80% / 90% 구간용 분위수
    q = {p: float(np.percentile(er, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
    print(f"\n  LOBO Er(%) (n={len(er)}): median={np.median(er):+.1f}, "
          f"mean={er.mean():+.1f}, std={er.std():.1f}")
    print("  → median>0 = 구조적 under-predict(early) = 2.5× late 페널티에 정합.")

    sub = pd.read_excel(SUB_XLSX).set_index("Bearing")["RUL_pred_seconds"]

    def act_from(pred, er_pct):
        # Act = Pred / (1 - Er/100); Er→100 방지 위해 clip
        denom = np.clip(1.0 - er_pct / 100.0, 0.15, None)
        return pred / denom

    rows = []
    for b, pred in sub.items():
        # 80% 구간: Er p10..p90, 90% 구간: p5..p95
        # actual 하한 = Er 큼(under 심함)일 때 Act 큼... 부호 주의:
        #   Er 큼(+) ⇒ pred << act ⇒ act 큼.  Er 작음(−) ⇒ pred >> act ⇒ act 작음.
        lo80 = act_from(pred, q[10]); hi80 = act_from(pred, q[90])
        lo90 = act_from(pred, q[5]);  hi90 = act_from(pred, q[95])
        rows.append({
            "Bearing": b, "point_pred_s": round(pred),
            "lo80_s": round(min(lo80, hi80)), "hi80_s": round(max(lo80, hi80)),
            "lo90_s": round(min(lo90, hi90)), "hi90_s": round(max(lo90, hi90)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "27_conformal_intervals.csv", index=False)

    print("\n  Test별 conformal 구간 (점추정은 제출본 불변):")
    print(f"  {'Bearing':8s} {'point(h)':>9} {'80%구간(h)':>20} {'90%구간(h)':>20}")
    for _, r in out.iterrows():
        print(f"  {r.Bearing:8s} {r.point_pred_s/3600:9.2f} "
              f"{r.lo80_s/3600:7.2f}~{r.hi80_s/3600:<7.2f}      "
              f"{r.lo90_s/3600:7.2f}~{r.hi90_s/3600:<7.2f}")
    print(f"\n  Saved: {RESULT_DIR / '27_conformal_intervals.csv'}")
    print("\n  [발표 포인트] 점추정 = asym-optimal, 구간 = LOBO 잔차 conformal.")
    print("  구간 폭이 큰 것은 train 4 베어링 한계의 정직한 반영 (과대 정밀 주장 회피).")


if __name__ == "__main__":
    main()
