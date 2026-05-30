"""36_BootstrapLOBO_CI — n=4 베어링 LOBO 점수의 부트스트랩 CI (방법 우열의 정직한 불확실성).

동기: iter46~54서 'physics-gated 0.592 = 최고' 등 점추정 우열을 주장. 그러나 train=4 베어링뿐 →
LOBO progression asym 평균의 표본분산이 큼. **4 베어링 부트스트랩**으로 (a) 각 방법 평균의 CI,
(b) physics-gated > avg-rate / > HI-only 의 승률 P를 산출 → 우열 주장이 노이즈 내인지 정직하게 경계.

대상(동일 FRACS 그리드에서 인과 재계산): HI-only-fit / avg-rate(32) / physics-gated(35).
(B per-bearing 선택은 별 인프라 필요 → 별도 그리드라 제외; 본 분석은 물리 방법 군 내 비교.)
결정론적: 부트스트랩 표본을 4 베어링의 모든 복원추출 조합(4^4=256)으로 **전수 열거**(Math.random 불가·재현성).

출력: artifacts/results/32_DegradationRate_RUL/36_bootstrap_lobo_ci.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys
import itertools

import numpy as np
import pandas as pd

ENSEMBLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENSEMBLE.parent))
from shared.utils import asym_score  # noqa: E402

FEAT = ENSEMBLE / "artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_features_dynamics.csv"
OUT = ENSEMBLE / "artifacts/results/32_DegradationRate_RUL/36_bootstrap_lobo_ci.csv"
TRAINS = ["Train1", "Train2", "Train3", "Train4"]
STEP_S = 600.0
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]


def hi_at(hi, i, w=3):
    lo = max(0, i - w + 1)
    return float(np.clip(np.median(hi[lo:i + 1]), 1e-3, 0.999))


def per_bearing_scores(df):
    """각 train 베어링의 progression 평균 asym (3 방법)."""
    # HI-only LOBO 전이용: 3개로 log(rul)~HI fit
    out = {m: {} for m in ["HI_only", "avg_rate", "phys_gated"]}
    # severity 임계(전체 train) + p_eol (physics-gated용)
    eol = df[df.rul_s <= 3000]
    te, tr = float(eol.energy_ratio.quantile(.9)), float(eol.rms_multi.quantile(.9))
    sev = df[(df.energy_ratio > te) | (df.rms_multi > tr) | (df.HI > 0.85)]
    grid = np.linspace(max(sev.rul_s.min(), 600), np.percentile(sev.rul_s, 95), 200)
    p_eol = float(grid[int(np.argmax([np.mean([asym_score(p, r) for r in sev.rul_s.values]) for p in grid]))])
    for held in TRAINS:
        others = df[df.bearing.isin([b for b in TRAINS if b != held])]
        b, a = np.polyfit(others.HI.values, np.log(np.clip(others.rul_s.values, 600, None)), 1)
        s = df[df.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
        sc = {m: [] for m in out}
        for f in FRACS:
            i = int(f * (len(s) - 1)); true = float(s.rul_s.iloc[i]); hi = hi_at(s.HI.values, i)
            sc["HI_only"].append(asym_score(max(np.exp(a + b * s.HI.iloc[i]), 600), true))
            sc["avg_rate"].append(asym_score(max(i * STEP_S * (1 - hi) / hi, 600), true))
            severe = (s.energy_ratio.iloc[i] > te) or (s.rms_multi.iloc[i] > tr) or (hi > 0.85)
            est = max(p_eol, 600) if severe else max(i * STEP_S * (1 - hi) / hi, 600)
            sc["phys_gated"].append(asym_score(est, true))
        for m in out:
            out[m][held] = float(np.mean(sc[m]))
    return out


def main():
    print("=" * 76)
    print("36_BootstrapLOBO_CI — n=4 베어링 LOBO 점수 부트스트랩 CI (우열의 정직한 불확실성)")
    print("=" * 76)
    df = df0 = pd.read_csv(FEAT)
    df = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s", "energy_ratio", "rms_multi"]].dropna()
    pb = per_bearing_scores(df)
    methods = ["HI_only", "avg_rate", "phys_gated"]
    print("\n  베어링별 LOBO progression asym:")
    print(f"  {'method':12} " + " ".join(f"{b:>8}" for b in TRAINS) + f"  {'mean':>7}")
    for m in methods:
        v = [pb[m][b] for b in TRAINS]
        print(f"  {m:12} " + " ".join(f"{x:8.3f}" for x in v) + f"  {np.mean(v):7.3f}")

    # 전수 부트스트랩: 4 베어링 복원추출 4^4=256 표본
    idx = list(itertools.product(range(4), repeat=4))
    boot = {m: np.array([np.mean([pb[m][TRAINS[j]] for j in s]) for s in idx]) for m in methods}
    rows = []
    print("\n  부트스트랩(256 표본) 평균 asym 95% CI:")
    for m in methods:
        lo, hi = np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)
        print(f"   {m:12} mean={boot[m].mean():.3f}  95%CI [{lo:.3f}, {hi:.3f}]")
        rows.append(dict(kind="ci", method=m, mean=round(boot[m].mean(), 3), lo=round(lo, 3), hi=round(hi, 3)))
    print("\n  pairwise 승률 P(A > B) (같은 부트스트랩 표본 페어링):")
    for a, b in [("phys_gated", "avg_rate"), ("phys_gated", "HI_only"), ("avg_rate", "HI_only")]:
        p = float(np.mean(boot[a] > boot[b]))
        print(f"   P({a} > {b}) = {p:.2f}")
        rows.append(dict(kind="winprob", method=f"{a}>{b}", mean=round(p, 2), lo="", hi=""))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print("\n  [해석] n=4라 CI가 넓으면 방법 간 우열은 통계적으로 약함 → 발표서 점추정 우열 과장 금지,")
    print("         '물리 방법군이 per-bearing/HI-prior와 동급~상위, 단 n=4 한계로 결정적 아님'으로 정직 서술.")


if __name__ == "__main__":
    main()
