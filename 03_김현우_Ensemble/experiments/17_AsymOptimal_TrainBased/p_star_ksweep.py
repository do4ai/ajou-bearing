"""40_PStarKSweep — flagship p* 의 유일 하이퍼파라미터 K(HI-KNN 이웃 수) 민감도 감사.

질문: p* 제출값(29377/27574/65379/32394/1200/41400)이 K=20 에 과의존하는가?
  K∈{8,12,16,20,28,40} 에서 (a) 6 테스트 p* 변동(셀별 변동계수 CV), (b) LOBO 전체 asym 변동을 측정.
  → 값/LOBO 가 K에 평탄하면 'K 단일 knob에 robust' = 합리성·발표 방어. 크게 흔들리면 fragile(6/8 전 인지 필요).

동일 production 엔진(p_star_estimator) 재사용 — 로직 불일치 없음. 점추정 불변(감사 전용).
산출: artifacts/results/17_AsymOptimal_TrainBased/40_pstar_ksweep.csv
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
from p_star_estimator import p_star, FEAT, RDIR, TRAINS, ORDER, FRACS  # noqa: E402
from shared.utils import asym_score  # noqa: E402  (paths set by p_star_estimator import)

KS = [8, 12, 16, 20, 28, 40]


def main() -> None:
    print("=" * 80)
    print("40_PStarKSweep — flagship p* 의 K 민감도 감사 (K∈%s)" % KS)
    print("=" * 80)

    df = pd.read_csv(FEAT)
    train = df[df.bearing.isin(TRAINS)][["bearing", "HI", "rul_s"]].dropna()
    last_hi = {b: float(df[df.bearing == b].HI.iloc[-1]) for b in ORDER}

    rows = []
    preds_by_k = {}
    lobo_by_k = {}
    for k in KS:
        preds = {b: p_star(last_hi[b], train, k)[0] for b in ORDER}
        preds_by_k[k] = preds
        # LOBO overall
        folds = []
        for held in TRAINS:
            others = train[train.bearing != held]
            s = train[train.bearing == held].sort_values("rul_s", ascending=False).reset_index(drop=True)
            scs = []
            for f in FRACS:
                i = int(f * (len(s) - 1))
                ps = p_star(float(s.HI.iloc[i]), others, k)[0]
                scs.append(asym_score(ps, float(s.rul_s.iloc[i])))
            folds.append(float(np.mean(scs)))
        lobo_by_k[k] = float(np.mean(folds))
        rows.append(dict(K=k, LOBO=round(lobo_by_k[k], 4),
                         **{b: round(preds[b]) for b in ORDER}))

    out = pd.DataFrame(rows)
    out.to_csv(RDIR / "40_pstar_ksweep.csv", index=False)
    print("\n  K | LOBO  | " + " ".join(f"{b:>7}" for b in ORDER))
    for r in rows:
        print(f"  {r['K']:2d}| {r['LOBO']:.4f}| " + " ".join(f"{r[b]:7d}" for b in ORDER))

    # 안정성 지표
    print("\n  [안정성] 셀별 K 변동(min~max, CV=std/mean):")
    for b in ORDER:
        vals = np.array([preds_by_k[k][b] for k in KS], dtype=float)
        cv = float(vals.std() / vals.mean()) if vals.mean() else 0.0
        print(f"    {b}: {vals.min():.0f}~{vals.max():.0f}  CV={cv:.3f}")
    lobos = np.array([lobo_by_k[k] for k in KS])

    # 카테고리(방향) 불변성 — 의사결정상 중요한 건 절대 % 아니라 'short/mid/long' 유지 여부.
    def cat(v: float) -> str:
        return "SHORT" if v < 5000 else ("LONG" if v > 15000 else "MID")
    cat_invariant = {}
    for b in ORDER:
        cats = {cat(preds_by_k[k][b]) for k in KS}
        cat_invariant[b] = (len(cats) == 1, next(iter(cats)) if len(cats) == 1 else "/".join(sorted(cats)))
    print(f"\n  LOBO 변동: {lobos.min():.4f}~{lobos.max():.4f} (range {lobos.max()-lobos.min():.4f})")
    print("  카테고리(방향) 불변성:")
    for b in ORDER:
        inv, lab = cat_invariant[b]
        vals = [preds_by_k[k][b] for k in KS]
        print(f"    {b}: {'불변' if inv else '변동'} = {lab:6s}  (값대역 {min(vals):.0f}~{max(vals):.0f})")

    all_cat_invariant = all(v[0] for v in cat_invariant.values())
    lobo_flat = (lobos.max() - lobos.min()) < 0.03
    print("\n  [판정]", end=" ")
    if lobo_flat and all_cat_invariant:
        print(f"K-ROBUST: LOBO 평탄(range {lobos.max()-lobos.min():.3f}) + 6 베어링 방향 전부 K-불변(K∈[8,40]).")
        print("           → K=20은 안정 평탄대. 점추정 방향(=의사결정)은 K에 의존하지 않음.")
        print("           정직 disclose: 절대 크기는 일부 셀(특히 T3 50k~73k) 변동 → K-sweep 대역을 민감도 구간으로 보고.")
    else:
        print(f"부분 민감: LOBO range {lobos.max()-lobos.min():.3f}, 방향변동 셀="
              + ",".join(b for b in ORDER if not cat_invariant[b][0]) + " → 발표서 명시.")
    print(f"\n  Saved: {RDIR / '40_pstar_ksweep.csv'}")


if __name__ == "__main__":
    main()
