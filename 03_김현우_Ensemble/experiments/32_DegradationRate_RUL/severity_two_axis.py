"""34_SeverityTwoAxis — 2축 건강평가(HI-level × energy-severity)로 Test 분류 (train-based).

iter46~47에서 HI 단독 열화모델이 Test6를 'long'으로 오판 → iter48서 다축 점검.
발견: HI는 '진행도', energy_ratio·rms_multi는 '심각도(severity)'를 담음. 둘은 분리될 수 있다.
  - 정상 mid-life: HI 중간 + energy/rms 정상 → 긴 RUL.
  - hidden-EOL: HI 중간이라도 energy/rms가 train EOL 수준 초과 → 짧은 RUL (숨은 급성 고장).

임계: train 'near-EOL'(rul≤3000)의 energy_ratio·rms_multi p90 (train 유래, 임의값 아님).
Test 각 베어링의 마지막 관측점을 분류 → 트랙 예측 방향 교차검증.

출력: artifacts/results/32_DegradationRate_RUL/34_severity_two_axis.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import pandas as pd

ENSEMBLE = Path(__file__).resolve().parents[2]
FEAT = ENSEMBLE / "artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_features_dynamics.csv"
OUT = ENSEMBLE / "artifacts/results/32_DegradationRate_RUL/34_severity_two_axis.csv"
TESTS = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]


def main() -> None:
    print("=" * 76)
    print("34_SeverityTwoAxis — HI-level × energy-severity 2축 분류 (train-based)")
    print("=" * 76)
    df = pd.read_csv(FEAT)
    tr = df[df.bearing.str.startswith("Train")]
    eol = tr[tr.rul_s <= 3000]

    # train EOL 유래 severity 임계 (p90)
    e_thr = float(eol.energy_ratio.quantile(0.90))
    r_thr = float(eol.rms_multi.quantile(0.90))
    print(f"\n  severity 임계 (train near-EOL p90): energy_ratio>{e_thr:.2f}, rms_multi>{r_thr:.2f}")

    PRED_B = {"Test1": 10067, "Test2": 10998, "Test3": 48900, "Test4": 9545, "Test5": 644, "Test6": 10275}
    PRED_A = {"Test1": 32035, "Test2": 33556, "Test3": 6449, "Test4": 14113, "Test5": 644, "Test6": 3000}

    rows = []
    print(f"\n  {'Test':7} {'HI':>6} {'energy':>8} {'rms':>6} {'severe?':>8} {'regime':>14}  (B / A)")
    for b in TESTS:
        r = df[df.bearing == b].reset_index(drop=True).iloc[-1]
        sev = (r.energy_ratio > e_thr) or (r.rms_multi > r_thr)
        # regime: HI 높음 OR severe → EOL(짧음); 아니면 mid/early(긺)
        eol_like = sev or (r.HI > 0.85)
        regime = "EOL(짧음)" if eol_like else "mid/early(긺)"
        rows.append(dict(bearing=b, HI=round(float(r.HI), 3),
                         energy_ratio=round(float(r.energy_ratio), 2), rms_multi=round(float(r.rms_multi), 3),
                         severe=bool(sev), regime=regime, predB=PRED_B[b], predA=PRED_A[b]))
        print(f"  {b:7} {r.HI:>6.3f} {r.energy_ratio:>8.2f} {r.rms_multi:>6.3f} {str(sev):>8} {regime:>14}  (B={PRED_B[b]} / A={PRED_A[b]})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print("\n  [결론] 2축 분류:")
    print("   - Test5/6 = EOL(짧음): Test5(HI+energy 둘다), Test6(energy/rms가 train EOL p90 초과=hidden severe).")
    print("     → 두 트랙의 Test6 짧은 예측(A3000/B10275) 데이터로 정당화. iter46/47 'Test6 long' 자체정정.")
    print("   - Test1/2/4 = mid-life(긺), energy 정상: B의 ~10k는 과소예측 가능 → A의 긴 예측이 정합적.")
    print("   - Test3 = early(긺), energy 정상: B 48900 정합, A 6449 과소.")


if __name__ == "__main__":
    main()
