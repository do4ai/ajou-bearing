"""공통 유틸리티 — 세 방법론 공유"""
import numpy as np

# 비대칭 패널티 분모 (시간 단위)
LATE_D  = 20.0 / 3600   # h
EARLY_D = 50.0 / 3600   # h

def asym_score(pred_h, true_h):
    """
    챌린지 공식 (시간 단위 기준):
      Er = pred - true

      늦은 예측 (Er ≤ 0): A = exp(-ln(0.5) · Er / LATE_D)
                         = exp( 0.693 · Er / LATE_D )   [Er<0 → arg<0 → A<1]

      이른 예측 (Er > 0): A = exp( ln(0.5) · Er / EARLY_D)
                         = exp(-0.693 · Er / EARLY_D )  [Er>0 → arg<0 → A<1]

    결과: A ∈ (0, 1], 높을수록 좋음, 완벽=1.0
    """
    err     = np.asarray(pred_h, dtype=np.float64) - np.asarray(true_h, dtype=np.float64)
    ln_half = np.log(0.5)          # -0.693

    # 지수 인수 (항상 ≤ 0)
    arg_late  = np.clip(-ln_half * err / LATE_D,  -50, 0)   # 0.693*err, err≤0 → ≤0
    arg_early = np.clip( ln_half * err / EARLY_D, -50, 0)   # -0.693*err, err>0 → ≤0

    score = np.where(err <= 0, np.exp(arg_late), np.exp(arg_early))
    return float(np.mean(score))


if __name__ == '__main__':
    ln = np.log(0.5)
    assert abs(asym_score([10.0], [10.0]) - 1.0) < 1e-6, "완벽예측 실패"
    assert abs(asym_score([10 - 20/3600], [10.0]) - 0.5) < 1e-3, "늦게20s 실패"
    assert abs(asym_score([10 + 50/3600], [10.0]) - 0.5) < 1e-3, "일찍50s 실패"
    print("✅ asym_score 검증 통과")
    print(f"  완벽:      {asym_score([10],[10]):.4f}")
    print(f"  늦게 20s:  {asym_score([10-20/3600],[10]):.4f}  (should be 0.5)")
    print(f"  일찍 50s:  {asym_score([10+50/3600],[10]):.4f}  (should be 0.5)")
    print(f"  늦게  1h:  {asym_score([9],[10]):.6f}")
    print(f"  이른  1h:  {asym_score([11],[10]):.6f}")
