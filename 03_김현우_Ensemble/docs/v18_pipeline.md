# 2_EOLDirect_4ch_WeightedRUL Pipeline (legacy v18)

## 목적

`2_EOLDirect_4ch_WeightedRUL`은 `1_Baseline_1ch_TFTBiLSTM_GPR` 대비 고장 직전(EOL) 예측을 강화한 실험이다. 공식 지표가 늦은 예측을 강하게 벌주기 때문에, 마지막 RUL 예측을 보수적으로 만들기 위해 설계되었다.

## 핵심 변경

- `rul_pw` 대신 실제 `rul_s` 직접 학습
- EOL 근처 샘플 가중치 대폭 강화
- 늦은 예측 페널티 강화
- 600초 예측 하한 적용
- early stopping에 Full score와 Last score를 함께 반영

## 입력 및 전처리

v17과 동일한 기본 전처리 구조를 사용한다.

1. ch0 중심 진동 피처 추출
2. Fast Kurtogram 스타일 대역 선택
3. Band-pass filtering
4. Hilbert envelope 생성
5. Envelope order energy 추출
6. DTC-VAE HI 생성

## 모델

- TFT 5 seeds
- BiLSTM 5 seeds
- `SEQ_LEN=10`
- fold별 scaler와 `rul_max` 저장

## 증강 및 불균형 보완

- Gaussian noise injection: `0.035`
- Mixup: `alpha=0.2`, `prob=0.3`
- EOL sample weighting
  - 가장 작은 RUL 5개: 50x
  - 다음 15개: 10x
  - 나머지: 1x

## Loss

- Weighted asymmetric MSE
- Official score loss
- 늦은 예측 페널티: v17보다 강화
- 학습 후반으로 갈수록 score loss 비중 증가

## 검증

- LOBO
- Full score: 전체 RUL 곡선 평균
- Last score: 마지막 측정 시점만 평가
- Combined score: `0.7 * Full + 0.3 * Last`

## 역할

`2_EOLDirect_4ch_WeightedRUL`은 `3_HIBlend_Baseline_EOLDirect`의 EOL-sensitive partner로 사용되었다. `5_HIBlend_Baseline_ChannelSym`에서는 이 모델 대신 `4_ChannelSym_EOLWeighted`가 사용되지만, `3_HIBlend_Baseline_EOLDirect` combined는 여전히 중요한 백업 제출 후보이다.
