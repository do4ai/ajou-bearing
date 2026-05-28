# 1_Baseline_1ch_TFTBiLSTM_GPR Pipeline (legacy v17)

## 목적

`1_Baseline_1ch_TFTBiLSTM_GPR`은 현재 앙상블의 안정적인 기본 모델이다. 전체 RUL 곡선과 일반적인 Validation 베어링 예측에서 강하며, `5_HIBlend_Baseline_ChannelSym`에서는 HI가 낮은 Test1, Test2, Test3, Test4, Test6의 주 예측원으로 사용된다.

## 입력 데이터

- Train: `Train1`~`Train4`, EOL까지 측정된 run-to-failure 데이터
- Validation/Test: `Test1`~`Test6`, EOL 미도달 상태까지 측정된 데이터
- 진동: 25.6 kHz, 4채널
- 측정 주기: 600초마다 1분 측정
- v17 주 입력: ch0 중심 진동 피처 + 4채널 통합 RMS 보조 피처

## 전처리

1. `load_bearing()`으로 `vibration.npy`, `operating.csv` 로드
2. ch0 진동 신호에서 기본 통계 추출
3. Fast Kurtogram 스타일 대역 탐색
   - 후보 대역: 0.5-2 kHz, 1-4 kHz, 2-6 kHz, 3-8 kHz, 4-10 kHz, 5-12 kHz
   - 각 대역의 kurtosis를 계산해 충격성이 큰 대역 선택
4. 선택 대역 또는 기본 1-6 kHz band-pass 적용
5. Hilbert transform으로 envelope 생성
6. Envelope spectrum에서 bearing order energy 계산

## 피처

- 시간 영역: `rms`, `std`, `kurtosis`, `skewness`, `peak`, `crest`, `p2p`, `shape_f`
- 다채널 보조: `rms_multi`
- Kurtogram: `fc`, `bw`, `sk_kurt`
- Envelope: `env_rms`, `env_kurt`
- 결함 order: `bpfi_e/snr/h_ratio`, `bpfo_e/snr/h_ratio`, `bsf_e/snr/h_ratio`, `ftf_e/snr/h_ratio`
- 운전조건: `rpm`, `torque`, `tf`, `tr`, `power_proxy`

## HI 생성

- DTC-VAE로 피처를 latent vector로 압축
- `latent_0`을 0~1 정규화해 `HI`로 사용
- monotonicity/trendability loss를 함께 사용해 시간이 지날수록 증가하는 건강지수 유도

## 라벨링

- CUSUM으로 FPT(열화 시작점) 탐지
- FPT 전: piecewise RUL을 평탄하게 유지
- FPT 후: EOL까지 선형 감소
- EOL은 마지막 측정 이후 600초로 가정

## 모델

- `TFTModel` 5 seeds: `[42, 7, 123, 2026, 99]`
- `BiLSTMModel` 5 seeds: `[365, 1234, 777, 5050, 9999]`
- 입력 sequence length: `SEQ_LEN=10`
- 각 sequence의 마지막 시점 RUL을 예측

## 증강 및 학습

- Gaussian noise injection: `AUG_NOISE_STD=0.035`
- Mixup: `alpha=0.2`, `prob=0.3`
- late-RUL sample weighting
  - RUL 하위 20%: weight 3.0
  - RUL 20~40%: weight 2.0
  - 나머지: weight 1.0
- Loss: weighted asymmetric MSE + official score loss
- 늦은 예측(Pred > Act)에 더 큰 페널티 적용

## 검증

- LOBO(Leave-One-Bearing-Out)
- 각 fold에서 Train 베어링 하나를 통째로 숨기고 나머지로 학습
- seed별 예측을 mean/median/trimmed로 결합
- `asym_score`로 Full/Last 계산

## 역할

v17은 v24의 robust baseline이다. HI가 낮거나 중간인 Validation 베어링에서는 v22보다 안정적이며, v24 combined 제출에서 대부분의 Test 예측값을 사실상 결정한다.
