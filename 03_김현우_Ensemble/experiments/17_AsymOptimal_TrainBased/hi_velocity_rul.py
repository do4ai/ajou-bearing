"""22_HIVelocity_RUL — HI 상승 속도 기반 EOL time 추정.

핵심 통찰:
  모든 Test는 정확히 50 측정 / 8.17h에서 잘림 (동일 wall-clock).
  HI만 다름 (0.165~0.944) → 같은 시간 동안 열화 속도가 다름.
  빨리 degrade (Test5 HI=0.944 in 8.17h) = 짧은 총 수명.
  느림 (Test3 HI=0.165 in 8.17h) = 긴 총 수명.

방법:
  1. Train 각 베어링에서 동일 시점 t=8.17h의 HI 관측 (early-window feature).
     실제로 Train은 8.17h 지점에 측정 존재 (10분 주기).
  2. 그 시점 (HI@8.17h, total_EOL_time) pair 수집.
  3. Test의 HI@8.17h → Train의 HI@8.17h와 매칭 → total EOL time → RUL.

차이 (vs 19_EOLProgression):
  - 19: 현재 HI에 매칭되는 train progression ratio 사용 (시점 무관)
  - 22: 동일 절대 시간(8.17h)에서의 HI 비교 (velocity-equivalent)
  → Test와 Train을 동일 관측 윈도우로 정렬 (apples-to-apples)

Outputs:
  artifacts/results/17_AsymOptimal_TrainBased/22_hi_velocity_rul.csv
  artifacts/results/17_AsymOptimal_TrainBased/22_hi_velocity_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("17_AsymOptimal_TrainBased")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"

TEST_WINDOW_H = 8.17       # Test 관측 종료 시점 (모든 Test 동일)
TEST_WINDOW_S = TEST_WINDOW_H * 3600


def asym_optimal_from_samples(rul_samples, min_pred=600.0):
    rul_samples = np.asarray(rul_samples, dtype=np.float64)
    max_pred = float(rul_samples.max() * 1.2 + 1000)
    def neg_obj(p):
        er = 100.0 * (rul_samples - p) / (rul_samples + 1e-12)
        ln_half = np.log(0.5)
        a = np.where(er <= 0, np.exp(np.clip(-ln_half * er / 20.0, -50, 0)),
                     np.exp(np.clip(ln_half * er / 50.0, -50, 0)))
        return -float(np.mean(a))
    res = minimize_scalar(neg_obj, bounds=(min_pred, max_pred), method="bounded")
    return float(res.x)


def hi_at_time(sub: pd.DataFrame, t_target_s: float) -> float:
    """베어링의 t_target_s 시점 HI (선형 보간)."""
    t = sub["t_s"].values
    hi = sub["HI"].values
    if t_target_s <= t[0]:
        return float(hi[0])
    if t_target_s >= t[-1]:
        return float(hi[-1])
    return float(np.interp(t_target_s, t, hi))


def main() -> None:
    print("=" * 70)
    print("22_HIVelocity_RUL — HI@8.17h 기반 EOL time 추정")
    print("=" * 70)

    df = pd.read_csv(FEATURE_CSV).fillna(0)

    # Train: (HI@8.17h, total_EOL_time) pair
    train_ref = []
    for nm in TRAIN_NAMES:
        sub = df[df.bearing == nm].sort_values("t_s").reset_index(drop=True)
        eol_t = float(sub["t_s"].iloc[-1]) + 600.0
        hi_at_window = hi_at_time(sub, TEST_WINDOW_S)
        train_ref.append({"bearing": nm, "hi_at_8h": hi_at_window,
                           "eol_time_s": eol_t, "eol_time_h": eol_t / 3600})
        print(f"  {nm}: HI@8.17h={hi_at_window:.3f}, EOL={eol_t/3600:.2f}h")
    ref_df = pd.DataFrame(train_ref)

    # 관계: HI@8.17h가 높을수록 EOL 짧음. 단조 관계 확인 + 회귀.
    print(f"\n  HI@8.17h vs EOL correlation:")
    corr = np.corrcoef(ref_df["hi_at_8h"], ref_df["eol_time_h"])[0, 1]
    print(f"    Pearson r = {corr:.3f} (음수 = 빠른 열화 → 짧은 수명)")

    # Test: HI@8.17h = HI_last (Test는 8.17h가 마지막)
    print(f"\n  Test predictions:")
    results = []
    for nm in VAL_NAMES:
        sub = df[df.bearing == nm].sort_values("t_s").reset_index(drop=True)
        hi_test = float(sub["HI"].iloc[-1])
        t_test = float(sub["t_s"].iloc[-1])

        # 방법1: weighted by HI 유사도 (가까운 train HI@8h 우선)
        weights = 1.0 / (np.abs(ref_df["hi_at_8h"].values - hi_test) + 0.05)
        eol_times = ref_df["eol_time_s"].values
        # weighted EOL → RUL
        est_ruls = np.maximum(eol_times - t_test, 600.0)
        pred_asym = asym_optimal_from_samples(est_ruls)
        # weighted mean EOL
        weol = float(np.sum(weights * eol_times) / np.sum(weights))
        pred_weighted = max(600.0, weol - t_test)
        # nearest single
        nearest_i = int(np.argmin(np.abs(ref_df["hi_at_8h"].values - hi_test)))
        pred_nearest = max(600.0, eol_times[nearest_i] - t_test)

        results.append({
            "Bearing": nm, "HI_last": hi_test,
            "HI_at_8h": hi_test,
            "pred_asym_optimal_s": pred_asym,
            "pred_weighted_s": pred_weighted,
            "pred_nearest_s": pred_nearest,
            "nearest_train": ref_df["bearing"].iloc[nearest_i],
            "nearest_eol_h": ref_df["eol_time_h"].iloc[nearest_i],
        })
        print(f"  {nm}: HI={hi_test:.3f}  asym={pred_asym:.0f}s  weighted={pred_weighted:.0f}s  "
              f"nearest={pred_nearest:.0f}s ({ref_df['bearing'].iloc[nearest_i]})")

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULT_DIR / "22_hi_velocity_rul.csv", index=False)
    sub = out_df[["Bearing", "HI_last", "pred_weighted_s"]].copy()
    sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "22_hi_velocity_submission.xlsx", index=False)

    # LOBO 검증: train의 한 베어링을 8.17h에서 자른 뒤 나머지로 추정
    print(f"\n[LOBO 검증]")
    lobo_rows = []
    for val in TRAIN_NAMES:
        ref = ref_df[ref_df.bearing != val]
        v_sub = df[df.bearing == val].sort_values("t_s").reset_index(drop=True)
        hi_v = hi_at_time(v_sub, TEST_WINDOW_S)
        # true RUL at 8.17h
        true_rul = max(600.0, (float(v_sub["t_s"].iloc[-1]) + 600.0) - TEST_WINDOW_S)
        weights = 1.0 / (np.abs(ref["hi_at_8h"].values - hi_v) + 0.05)
        weol = float(np.sum(weights * ref["eol_time_s"].values) / np.sum(weights))
        pred = max(600.0, weol - TEST_WINDOW_S)
        score = asym_score([pred], [true_rul])
        lobo_rows.append({"val": val, "hi_at_8h": hi_v, "true_rul": true_rul,
                           "pred": pred, "score": score})
        print(f"  {val}: HI@8h={hi_v:.3f}  true_rul={true_rul:.0f}  pred={pred:.0f}  score={score:.3f}")
    lobo_df = pd.DataFrame(lobo_rows)
    lobo_df.to_csv(RESULT_DIR / "22_hi_velocity_lobo.csv", index=False)
    print(f"\n  LOBO mean score: {lobo_df['score'].mean():.4f}")
    print(f"\n  Saved: {RESULT_DIR / '22_hi_velocity_rul.csv'}")


if __name__ == "__main__":
    main()
