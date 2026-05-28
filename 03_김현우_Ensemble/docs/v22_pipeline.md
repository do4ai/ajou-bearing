# 4_ChannelSym_EOLWeighted Pipeline (legacy v22)

## 목적

`4_ChannelSym_EOLWeighted`는 `2_EOLDirect_4ch_WeightedRUL`의 EOL 민감 학습 설정과 4채널/채널 대칭 피처를 결합한 모델이다. 고장 직전 신호를 더 민감하게 감지하고, 특정 채널 고장 패턴을 놓치지 않는 것이 목적이다.

## 입력

- 4채널 진동 전체
- Train 운전조건: rpm, torque, temp_front, temp_rear
- Validation/Test 운전조건은 현재 변환 결과에서 결측이므로 기본값으로 대체됨

## 전처리

1. 각 채널별 기본 통계 추출
2. 각 채널별 Fast Kurtogram 스타일 대역 선택
3. 각 채널별 band-pass filtering
4. 각 채널별 Hilbert envelope 계산
5. 각 채널별 envelope order energy 추출
6. 4채널 대칭 피처 생성

## 채널 대칭 피처

각 공통 피처에 대해 다음을 계산한다.

- `chsym_max_*`
- `chsym_min_*`
- `chsym_range_*`
- `chsym_std_*`
- `chsym_top2_*`

추가 다채널 관계 피처:

- 채널 간 correlation: `corr_01` 등
- `energy_max`, `energy_min`, `energy_ratio`, `energy_std`

## HI 생성

- v22 전용 DTC-VAE 사용
- input dimension 증가에 맞춰 encoder/decoder 확장
- `latent_0`을 0~1 정규화해 HI로 사용

## 라벨 및 학습

- `rul_s` 직접 학습
- 예측값 600초 하한 적용
- EOL 샘플 weighting
  - 가장 작은 RUL 5개: 50x
  - 다음 15개: 10x
- 늦은 예측 페널티: `ASYM_PENALTY=5.0`
- Loss: weighted asym MSE + score loss

## 모델

- TFT 5 seeds
- BiLSTM 5 seeds
- `SEQ_LEN=10`
- seed ensemble: mean/median/trimmed/cons/ultra 후보 산출

## 검증 결과

`artifacts/results/04_ChannelSym_EOLWeighted/lobo_v22.csv` 기준:

- Train1 full median: 0.7668, last median: 1.0000
- Train2 full median: 0.7048, last median: 1.0000
- Train3 full median: 0.5055, last median: 1.0000
- Train4 full median: 0.3394, last median: 1.0000

## 역할

`4_ChannelSym_EOLWeighted`는 Test5에서 `HI_last=0.944`, `RUL=600s`를 제시하면서 `5_HIBlend_Baseline_ChannelSym`의 EOL-sensitive branch 역할을 한다.
