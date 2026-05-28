# 실험 요약 — KSPHM-KIMM 2026 RUL Prediction

> 앞으로 실험명은 `번호_방법명`을 기본으로 사용한다.
> `v17`, `v22` 같은 legacy ID는 코드/모델 경로 추적용으로만 괄호 안에 남긴다.

## 실험 이름 한눈에

| 새 이름 | Legacy ID | 입력 | 모델/방법 | 핵심 변경점 |
|---------|-----------|------|-----------|-------------|
| **1_Baseline_1ch_TFTBiLSTM_GPR** | v17 | 1ch, 31 feats, `rul_pw` | TFTx5 + BiLSTMx5 + GPR sigma | 최초 baseline, mid-life 안정 |
| **2_EOLDirect_4ch_WeightedRUL** | v18 | 4ch, 116 feats, `rul_s` | TFTx5 + BiLSTMx5 | RUL seconds 직접 학습, EOL 50x/10x 가중, asym penalty |
| **3_HIBlend_Baseline_EOLDirect** | v19 | `1` + `2` 출력 + HI | HI sigmoid weighted blend | low HI는 baseline, high HI는 EOLDirect |
| **4_ChannelSym_EOLWeighted** | v22 | 4ch, 231 feats, `rul_s` | TFTx5 + BiLSTMx5 | channel symmetry features 추가 |
| **5_HIBlend_Baseline_ChannelSym** | v24 | `1` + `4` 출력 + HI | HI sigmoid weighted blend | 현재 안정 best 제출 계열 |
| **6_Dynamics_DTW_TFTBiLSTM** | v25 | 4ch, 271 feats, `rul_s` | TFTx3 + BiLSTMx3 | dynamics features + DTW sanity |
| **7_DomainAdv_Dynamics_TFT** | v26 | 271 feats | TFTx3 + domain head + stage head | multi-source domain adversarial alignment |
| **8_HIBlend_Baseline_Dynamics** | blend | `1` + `6` 출력 + HI | HI sigmoid weighted blend | dynamics 모델과 baseline blend |
| **9_HIBlend_Baseline_DomainAdv** | blend | `1` + `7` 출력 + HI | HI sigmoid weighted blend | domain-adversarial 모델과 baseline blend |

## LOBO 4-fold 결과

| 새 이름 | Full | Last | Combined | 비고 |
|---------|------|------|----------|------|
| 1_Baseline_1ch_TFTBiLSTM_GPR | — | 0.119 | — | `rul_s` 기준 last가 매우 약함 |
| 2_EOLDirect_4ch_WeightedRUL | 0.558 | 0.956 | 0.677 | last 극적 개선 |
| 3_HIBlend_Baseline_EOLDirect | 0.603 | 0.978 | 0.715 | last 안정성이 좋음 |
| 4_ChannelSym_EOLWeighted | 0.579 | 1.000 | 0.705 | last perfect, full은 약간 손해 |
| **5_HIBlend_Baseline_ChannelSym** | **0.599** | **0.975** | **0.712** | 현재 안정 best 제출 후보 |
| 6_Dynamics_DTW_TFTBiLSTM | 0.544 | 0.750 | 0.605 | Train3 fold last 폭주, 14,314s |
| 7_DomainAdv_Dynamics_TFT | 0.602 | 0.750 | 0.646 | full은 좋지만 Train3 fold last 폭주, 6,313s |
| 8_HIBlend_Baseline_Dynamics | 0.588 | 0.725 | 0.636 | `5_HIBlend`를 못 이김 |
| 9_HIBlend_Baseline_DomainAdv | 0.626 | ~0.750 | ~0.72 | full 기준 최신 최고, Test5 EOL 리스크 큼 |

## 핵심 의사결정 흔적

### 1_Baseline → 2_EOLDirect

- `1_Baseline`은 piecewise RUL을 타겟으로 사용했는데, 실제 평가는 마지막 시점 `rul_s` 기준이라 last가 매우 약했다.
- `2_EOLDirect`에서 `rul_s` 직접 학습, 마지막 5측정 50x, 다음 15측정 10x, asym penalty 5x를 적용했다.
- LOBO Last가 0.119에서 0.956으로 크게 상승했다.

### 2_EOLDirect → 4_ChannelSym

- Train4는 ch3에서만 RMS/kurtosis가 급등하고 다른 채널은 잔잔했다.
- 각 채널 통계의 `max/min/range/std/top2`를 추가해 한 채널 이상만으로도 고장을 잡게 했다.
- 결과적으로 Last 1.000을 달성했지만 Full은 0.558에서 0.543~0.579 범위로 약간 손해가 있었다.

### 4_ChannelSym → 5_HIBlend_Baseline_ChannelSym

- `1_Baseline`은 mid-life에서 안정적이고, `4_ChannelSym`은 EOL 직전에서 정확했다.
- HI sigmoid로 역할을 분리했다. HI가 낮으면 `1_Baseline`, HI가 높으면 `4_ChannelSym`을 사용한다.
- Test5는 HI 0.944라 EOL 모델 쪽으로 강하게 이동해 644s를 예측한다.

### 5_HIBlend → 6_Dynamics

- 한 시점의 절대값보다 변화 속도가 더 신뢰 가능하다는 가설로 dynamics features를 추가했다.
- HI/RMS/EnvKurt/EnergyRatio/ChsymMaxKurt 계열에 `d1/d3/d5/slope5/slope10/acc/roll_std5`를 붙였다.
- DTW sanity에서 Test5는 Train1 RUL 1200s 근처와 유사해 EOL 판단을 유지할 근거가 생겼다.
- 다만 Train3 last가 14,314s로 폭주해 직접 제출/주력 blend로는 부적합하다.

### 6_Dynamics → 7_DomainAdv

- Train1~4와 Test 분포 차이를 줄이기 위해 domain classifier + gradient reversal을 붙였다.
- HI-stage CE를 추가해 degradation stage-aware alignment를 시도했다.
- Full mean은 좋아졌지만 Train3 last 폭주가 남아 EOL 영역에서는 위험하다.

### 8/9_HIBlend 계열

- `8_HIBlend_Baseline_Dynamics`는 `5_HIBlend`보다 나빴다.
- `9_HIBlend_Baseline_DomainAdv`는 full mean 기준 가장 높지만, Test5에서 `7_DomainAdv`가 12,489s를 예측해 EOL 리스크가 크다.
- 따라서 `9`는 `13_EOLHazardGate_Calibrator` 없이는 최종 제출 후보로 쓰지 않는다.

## 폐기된 실험

| 새 이름 | Legacy ID | 사유 | 위치 |
|---------|-----------|------|------|
| **10_EOLWeightAblation_LowerWeights** | v20 | EOL weight 50x를 완화했더니 Last 0.500 수준으로 붕괴 | `legacy/archive_failed/pipelines/pipeline_v20.py` |
| **11_ThreeWayBlend_Baseline_EOLDirect_Ablation** | v21 | `1+2+10` 3-way blend. `3_HIBlend`보다 나쁨 | `legacy/archive_failed/blends/blend_v17_v18_v20.py` |
| **12_ThreeWayBlend_Baseline_EOLDirect_ChannelSym** | v23 | `1+2+4` 3-way blend. `5_HIBlend`보다 나쁨 | `legacy/archive_failed/blends/blend_v17_v18_v22.py` |

## 다음 작업 이름

| 새 이름 | 역할 | 성공 기준 |
|---------|------|-----------|
| **13_EOLHazardGate_Calibrator** | EOL 위험확률 기반 clamp/gate | Train3/Test5/Test6 폭주 방지, worst-last 개선 |
| **14_RPMAwareOrderFeatures** | rpm 기반 fault order feature 재계산 | BPFI/BPFO/BSF/FTF 물리축 보정 후 LOBO 개선 |
| **15_TrajectoryKNN_DTW_RUL** | nearest degradation trajectory 기반 RUL 분포 | Test5/Test6의 RUL 근거 제공 |
| **16_ScoreAware_CalibratedEnsemble** | official asym score 기반 최종 calibration | last/worst-case 중심 최종 제출 생성 |

## 제출 후보

| 순위 | 파일 | 전략 | 주의사항 |
|------|------|------|----------|
| 1 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_balanced_submission.xlsx` | `16_ScoreAware_CalibratedEnsemble` balanced | 현재 추천 비교 후보. Test6 6000s clamp |
| 2 | `artifacts/results/05_HIBlend_Baseline_ChannelSym/submission_v24_v17v22_combined.xlsx` | `5_HIBlend_Baseline_ChannelSym` combined | 기존 안정 best. Test6 clamp 없음 |
| 3 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_safe_submission.xlsx` | `16_ScoreAware_CalibratedEnsemble` safe | Test4/Test6 보수 후보 |
| 보류 | `artifacts/results/09_HIBlend_Baseline_DomainAdv/submission_v9_v17v26_*.xlsx` | `9_HIBlend_Baseline_DomainAdv` | EOL gate 없이 단독 제출 금지 |

## 다음 실행 순서

1. `13_EOLHazardGate_Calibrator` 생성.
2. `14_RPMAwareOrderFeatures`로 order feature 재계산.
3. `15_TrajectoryKNN_DTW_RUL`로 Test별 유사 RUL 분포 산출.
4. `16_ScoreAware_CalibratedEnsemble`로 safe/balanced/aggressive 제출 3종 생성.
