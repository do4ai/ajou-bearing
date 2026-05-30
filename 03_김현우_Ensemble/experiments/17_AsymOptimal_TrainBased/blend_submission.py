"""43_BlendSubmission — flagship 앙상블 제출본 재현 스크립트 (code.zip 단독 재현 보장).

flagship = 두 독립 train-based 추정기의 베어링별 고정 기하평균:
  · p* (37_pstar_submission.xlsx, p_star_estimator.py 산출)
  · avg-rate 물리 (32_degradation_rate_rul.csv 의 test rul_est_avg, 32/predict.py 산출)
  → geomean → 42_blend_submission.xlsx (= HUFS_validation_blend.xlsx).
부수로 32_avgrate_submission.xlsx(정확도 anchor 제출본)도 재생성 — 둘 다 인라인 생성이라
그동안 재현 스크립트가 없었음(우수성/합리성/재현성 gap) → 본 스크립트로 폐쇄.

0-param(학습 weight 無). 점추정 불변. 산출 검증: preflight source bit-exact.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENSEMBLE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (HERE, ENSEMBLE, ENSEMBLE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from p_star_estimator import p_star, FEAT  # noqa: E402 (p* 엔진 재사용 — zip 동봉)

R17 = ENSEMBLE / "artifacts/results/17_AsymOptimal_TrainBased"
R32 = ENSEMBLE / "artifacts/results/32_DegradationRate_RUL"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
ORDER = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
FLOOR = 600.0
STEP_S = 600.0


def _sub_frame(bearings, secs) -> pd.DataFrame:
    df = pd.DataFrame({"Bearing": bearings, "RUL_pred_seconds": np.asarray(secs, float)})
    df["RUL_pred_hours"] = df["RUL_pred_seconds"] / 3600.0
    return df


def _avg_rate_test(s_hi: np.ndarray) -> float:
    """avg-rate 물리 (32/predict.py 규약): hi_now=최근3 중앙값(인과), elapsed=(n-1)*600."""
    i = len(s_hi) - 1
    hi_now = max(float(np.median(s_hi[max(0, i - 2):i + 1])), 1e-3)
    elapsed = i * STEP_S
    return max(elapsed * (1.0 - hi_now) / hi_now, FLOOR)


def main() -> None:
    print("=" * 72)
    print("43_BlendSubmission — flagship 앙상블(avg-rate × p*) 제출본 재현 (FEAT 단독·self-contained)")
    print("=" * 72)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s"]].dropna()

    ps_v, av_v = [], []
    for b in ORDER:
        s = df[df.bearing == b]
        hi_last = float(s.HI.iloc[-1])
        ps_v.append(p_star(hi_last, train)[0])                 # p* anchor (HI-KNN argmax E[asym])
        av_v.append(_avg_rate_test(s.HI.values))               # avg-rate anchor (물리)
    ps_v, av_v = np.array(ps_v), np.array(av_v)
    blend_v = np.sqrt(np.maximum(ps_v, FLOOR) * np.maximum(av_v, FLOOR))   # 고정 기하평균 (0-param)

    R17.mkdir(parents=True, exist_ok=True)
    R32.mkdir(parents=True, exist_ok=True)
    _sub_frame(ORDER, av_v).to_excel(R32 / "32_avgrate_submission.xlsx", index=False)
    _sub_frame(ORDER, blend_v).to_excel(R17 / "42_blend_submission.xlsx", index=False)

    comp = pd.DataFrame({"Bearing": ORDER, "p_star": ps_v.round(),
                         "avg_rate": av_v.round(), "blend_geo": blend_v.round()})
    print("\n" + comp.to_string(index=False))
    print(f"\n  재생성: 32_avgrate_submission.xlsx · 42_blend_submission.xlsx (FEAT + p_star_estimator 만으로)")
    print(f"  (제출 사본: HUFS_validation_avgrate.xlsx · HUFS_validation_blend.xlsx — preflight source bit-exact 대조)")


if __name__ == "__main__":
    main()
