# 5_HIBlend_Baseline_ChannelSym Pipeline (legacy v24)

## 목적

`5_HIBlend_Baseline_ChannelSym`은 안정적인 `1_Baseline_1ch_TFTBiLSTM_GPR`과 EOL 민감한 `4_ChannelSym_EOLWeighted`를 HI 기준으로 섞은 최종 제출 후보이다. HI가 낮은 베어링은 baseline을 신뢰하고, HI가 높은 베어링은 channel symmetry EOL 모델을 신뢰한다.

## 구성 모델

- `1_Baseline_1ch_TFTBiLSTM_GPR`: 안정적인 기본 모델, 중간 수명/일반 패턴에 강함
- `4_ChannelSym_EOLWeighted`: 4채널 대칭 피처와 EOL 가중치 기반, 고장 직전 감지에 강함

## 피처 및 HI

- `1_Baseline` 피처와 `4_ChannelSym` 피처를 각각 별도 추출
- `5_HIBlend` 블렌딩에는 `4_ChannelSym`의 4채널 기반 HI를 사용
- 이유: 채널별 고장 신호를 더 잘 반영하기 위함

## OOF 예측

LOBO fold별로 다음을 계산한다.

1. 숨긴 Train 베어링에 대해 `1_Baseline` fold 모델 예측
2. 같은 베어링에 대해 `4_ChannelSym` fold 모델 예측
3. 각 모델 seed 예측의 median 사용
4. v22 HI를 기준으로 blending weight 계산

## Blending 공식

```text
w_high = sigmoid((HI - threshold) * slope)
w_low  = 1 - w_high
pred   = 1_Baseline * w_low + 4_ChannelSym * w_high
pred   = max(pred * beta, 600)
```

## Grid Search

- `threshold`: 0.3~0.9
- `slope`: 3, 5, 10, 20, 40
- `beta`: 0.85~1.10
- 평가: Full, Last, Combined
- Combined = `0.7 * Full + 0.3 * Last`

## 주요 결과

| 후보 | Full | Last | Combined |
|---|---:|---:|---:|
| `5_HIBlend` best full | 0.637 | 0.308 | 0.538 |
| `5_HIBlend` best combined | 0.599 | 0.975 | 0.712 |

## 최종 Validation 예측

| Bearing | HI_last | 1_Baseline | 4_ChannelSym | 5_HIBlend combined |
|---|---:|---:|---:|---:|
| Test1 | 0.463 | 32,035s | 39,775s | 32,035s |
| Test2 | 0.500 | 33,556s | 39,303s | 33,556s |
| Test3 | 0.165 | 6,449s | 7,311s | 6,449s |
| Test4 | 0.448 | 14,113s | 9,818s | 14,113s |
| Test5 | 0.944 | 14,645s | 600s | 644s |
| Test6 | 0.410 | 11,641s | 881s | 11,641s |

## 판단

`5_HIBlend_Baseline_ChannelSym` combined는 현재 제출 후보 1순위이다. 단, Test5가 실제로 EOL 근접인지가 가장 큰 리스크이다. 따라서 `6_Dynamics_DTW_TFTBiLSTM`에서는 Test5의 높은 HI가 실제 열화 가속인지 DTW/동역학 피처로 검증한다.
