# 6_Dynamics_DTW_TFTBiLSTM Pipeline (legacy v25)

## 위치 (압축 ID)

- 공식 이름: **6_Dynamics_DTW_TFTBiLSTM**
- Legacy ID: **v25**
- 베이스: **4_ChannelSym_EOLWeighted** (= legacy v22, EOL-focus + 채널 대칭)
- 추가: 동역학 피처 (35 dims) + DTW 궤적 sanity check

## 목적

`5_HIBlend_Baseline_ChannelSym`의 핵심 리스크인 Test5 판단을 강화한다. `4_ChannelSym_EOLWeighted`까지는 "현재 시점 피처"만 사용해 HI 한 점에 과민할 수 있었다. `6_Dynamics_DTW_TFTBiLSTM`은 HI/RMS/EnvKurt/EnergyRatio/ChsymMaxKurt 다섯 base feature에 대해 **d/slope/acc/roll_std** 시간 미분 피처를 더해, **열화 속도와 가속도**를 모델이 학습하도록 한다.

## 새 입력 피처 (35 dims)

base × dynamics = 5 × 7 = 35

base = [HI, rms_multi, energy_ratio, chsym_max_kurt, chsym_max_env_kurt]
dyn  = [d1, d3, d5, slope5, slope10, acc, roll_std5]

총 피처 = v22(231) + dynamics(35) + latent(4) + HI(1) = 271

## 모델 / 앙상블

- TFT (TCN-Transformer) × 3 seeds + BiLSTM × 3 seeds (총 6 seeds)
- SEQ_LEN = 10, EPOCHS = 200, ASYM_PENALTY = 5.0
- EOL 가중치 50× (마지막 5측정), 10× (다음 15측정)
- Loss = α · weighted_asym_MSE + (1-α) · score_loss
- 600s 하한 클립

## DTW Sanity (별도)

`pipeline_v25_dynamics_dtw.py` — 학습은 안 하고 다음만 계산:
- 각 Test 베어링의 마지막 50개 측정 시계열 ↔ 각 Train 베어링의 모든 50개 sliding window
- feature trajectory: [HI, rms_multi, energy_ratio, chsym_max_env_kurt, HI_slope5, HI_d5]
- 가장 유사한 Train window의 그 시점 RUL 반환
- 결과: `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_dynamics_dtw.csv`, `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_test5_sanity.csv`

### DTW 핵심 결과

| Bearing | 가장 유사한 Train idx | 그 시점 RUL | HI_last | HI_slope5 | HI_d5 |
|---------|----------------------|-------------|---------|-----------|-------|
| Test1 | Train2 idx=49 | 39,000s | 0.463 | -0.428 | -0.009 |
| Test2 | Train1 idx=59 | 40,200s | 0.500 | +0.572 | +0.303 |
| Test3 | Train2 idx=65 | 29,400s | 0.165 | -0.278 | +0.055 |
| Test4 | Train2 idx=49 | 39,000s | 0.448 | -0.767 | -0.499 |
| **Test5** | **Train1 idx=124** | **1,200s** | **0.944** | **-0.349** | **-0.020** |
| Test6 | Train2 idx=79 | 21,000s | 0.410 | +0.056 | +0.088 |

> Test5의 HI=0.944는 높지만 HI_slope5가 음수 → 최근 HI는 오히려 감소. DTW로는 Train1의 EOL 직전(idx 124, rul 1200s)과 가장 닮음.

## 산출물

- `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/lobo_v25.csv` — Train1~4 LOBO 점수
- `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_features_dynamics.csv` — 피처 (Train+Test 통합, 271 cols)
- `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_Train{1..4}.png` — fold별 예측 곡선
- `artifacts/models/06_Dynamics_DTW_TFTBiLSTM/fold_Train*/` — 모델 가중치 (tft × 3, bilstm × 3)

## 다음 (7_DomainAdv_Dynamics_TFT)

HI-stage pseudo label (early/mid/late) + multi-source domain adversarial을 `6_Dynamics_DTW_TFTBiLSTM` 위에 얹어 Train↔Test feature 분포 차이를 줄인다.
