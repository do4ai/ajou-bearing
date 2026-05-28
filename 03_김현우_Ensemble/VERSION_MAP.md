# Method Name Map — 03 김현우 Ensemble

> 혼동 방지를 위해 앞으로 문서/대화에서는 `v17`, `v25` 같은 실험 ID 대신 아래의 `번호_방법명`을 기본 이름으로 사용한다.
> 기존 ID는 파일/모델 경로 추적용 legacy ID로만 남긴다.

## 이름 규칙

- 형식: `번호_핵심방법_핵심모델또는목적`
- 번호는 실험이 진행된 계보 순서에 맞춰 고정한다.
- `.py` 파일명과 모델 폴더명은 import/저장 경로 호환을 위해 당장 변경하지 않는다.
- 제출/보고서에서는 `1_Baseline_1ch_TFTBiLSTM`처럼 방법명이 먼저 보이게 작성한다.

## 의미있는 실험 이름

| 새 이름 | Legacy ID | 모델/방법 요약 | Full | Last | Combined | 주요 파일/폴더 |
|---------|-----------|----------------|------|------|----------|----------------|
| **1_Baseline_1ch_TFTBiLSTM_GPR** | v17 / old v1 | 1ch 31 feats, TFTx5 + BiLSTMx5, GPR sigma, piecewise RUL | — | 0.119 | — | `experiments/01_Baseline_1ch_TFTBiLSTM_GPR/` |
| **2_EOLDirect_4ch_WeightedRUL** | v18 / old v2 | 4ch 116 feats, RUL seconds 직접 학습, EOL 50x/10x 가중, asym loss | 0.558 | 0.956 | 0.677 | `experiments/02_EOLDirect_4ch_WeightedRUL/` |
| **3_HIBlend_Baseline_EOLDirect** | v19 / old v3 | `1_Baseline` + `2_EOLDirect`, HI sigmoid weighted blend | 0.603 | 0.978 | 0.715 | `experiments/03_HIBlend_Baseline_EOLDirect/` |
| **4_ChannelSym_EOLWeighted** | v22 / old v4 | `2_EOLDirect` + channel symmetry features, 4ch 231 feats | 0.579 | **1.000** | 0.705 | `experiments/04_ChannelSym_EOLWeighted/` |
| **5_HIBlend_Baseline_ChannelSym** | v24 / old v5 | `1_Baseline` + `4_ChannelSym`, HI sigmoid blend, 현재 best combined 계열 | 0.599 | 0.975 | **0.712** | `experiments/05_HIBlend_Baseline_ChannelSym/` |
| **6_Dynamics_DTW_TFTBiLSTM** | v25 / old v6 | `4_ChannelSym` + dynamics features + DTW sanity, TFTx3 + BiLSTMx3 | 0.544 | 0.750 | 0.605 | `experiments/06_Dynamics_DTW_TFTBiLSTM/` |
| **7_DomainAdv_Dynamics_TFT** | v26 / old v7 | `6_Dynamics` + multi-source domain adversarial + HI-stage CE | **0.602** | 0.750 | 0.646 | `experiments/07_DomainAdv_Dynamics_TFT/` |
| **8_HIBlend_Baseline_Dynamics** | blend old v8 | `1_Baseline` + `6_Dynamics`, HI sigmoid blend | 0.588 | 0.725 | 0.636 | `experiments/08_HIBlend_Baseline_Dynamics/` |
| **9_HIBlend_Baseline_DomainAdv** | blend old v9 | `1_Baseline` + `7_DomainAdv`, HI sigmoid blend, full 기준 최신 최고 | 0.626 | ~0.750 | ~0.72 | `experiments/09_HIBlend_Baseline_DomainAdv/` |

## 폐기/비추천 실험 이름

| 새 이름 | Legacy ID | 폐기 사유 | 위치 |
|---------|-----------|----------|------|
| **10_EOLWeightAblation_LowerWeights** | v20 | EOL weight 50x를 30x/5x/1x로 완화했더니 last 성능 붕괴 | `legacy/archive_failed/pipelines/pipeline_v20.py`, `legacy/archive_failed/models/models_v20/` |
| **11_ThreeWayBlend_Baseline_EOLDirect_Ablation** | v21 | `1+2+10` 3-way blend. `3_HIBlend_Baseline_EOLDirect`보다 나쁨 | `legacy/archive_failed/blends/blend_v17_v18_v20.py` |
| **12_ThreeWayBlend_Baseline_EOLDirect_ChannelSym** | v23 | `1+2+4` 3-way blend. `5_HIBlend_Baseline_ChannelSym`보다 나쁨 | `legacy/archive_failed/blends/blend_v17_v18_v22.py` |

## 다음 실험용 예약 이름

| 예약 이름 | 목적 | 비고 |
|-----------|------|------|
| **13_EOLHazardGate_Calibrator** | Train3/Test5/Test6 같은 EOL 폭주 방지용 gate/clamp | 새 최우선 실험 |
| **14_RPMAwareOrderFeatures** | rpm 기반 BPFI/BPFO/BSF/FTF order feature 재계산 | 물리 feature 보정 |
| **15_TrajectoryKNN_DTW_RUL** | window similarity/DTW 기반 RUL 분포 추정 | 모델 출력 검증/보정 |
| **16_ScoreAware_CalibratedEnsemble** | official asym score와 worst-case 기반 최종 calibrated ensemble | 최종 제출 후보 생성 |

## 계보

```text
1_Baseline_1ch_TFTBiLSTM_GPR
  └─ 2_EOLDirect_4ch_WeightedRUL
       ├─ 3_HIBlend_Baseline_EOLDirect
       └─ 4_ChannelSym_EOLWeighted
            ├─ 5_HIBlend_Baseline_ChannelSym  ← 현재 안정 best
            └─ 6_Dynamics_DTW_TFTBiLSTM
                 ├─ 7_DomainAdv_Dynamics_TFT
                 ├─ 8_HIBlend_Baseline_Dynamics
                 └─ 9_HIBlend_Baseline_DomainAdv

폐기:
10_EOLWeightAblation_LowerWeights
  ├─ 11_ThreeWayBlend_Baseline_EOLDirect_Ablation
  └─ 12_ThreeWayBlend_Baseline_EOLDirect_ChannelSym

다음:
13_EOLHazardGate_Calibrator
14_RPMAwareOrderFeatures
15_TrajectoryKNN_DTW_RUL
16_ScoreAware_CalibratedEnsemble
```

## 핵심 의사결정 요약

- **1_Baseline → 2_EOLDirect**: piecewise RUL 대신 `rul_s` 직접 학습. last가 0.119에서 0.956으로 크게 개선.
- **2_EOLDirect → 4_ChannelSym**: 채널 대칭 피처로 한 채널만 이상해지는 Train4류 상황을 포착. last 1.000 달성.
- **4_ChannelSym → 5_HIBlend_Baseline_ChannelSym**: mid-life는 `1_Baseline`, EOL은 `4_ChannelSym`이 담당하도록 HI sigmoid로 분리. 현재 안정 best.
- **5_HIBlend → 6_Dynamics/7_DomainAdv**: dynamics와 domain adversarial은 full에는 도움 가능하지만 Train3 last 폭주가 있어 EOL 영역 직접 사용 금지.
- **3-way blend 계열(11, 12)**: 단순 모델 다양성보다 명확한 역할 분담이 더 효과적이어서 폐기.

## Test 처리 흐름 (5_HIBlend_Baseline_ChannelSym 기준)

| Bearing | HI_last | 1_Baseline | 4_ChannelSym | 5_HIBlend 최종 | 설명 |
|---------|---------|------------|--------------|----------------|------|
| Test1 | 0.463 | 32,035s | 39,775s | 32,035s | low/mid HI, baseline 사용 |
| Test2 | 0.500 | 33,556s | 39,303s | 33,556s | low/mid HI, baseline 사용 |
| Test3 | 0.165 | 6,449s | 7,311s | 6,449s | low HI, baseline 사용 |
| Test4 | 0.448 | 14,113s | 9,818s | 14,113s | low/mid HI, baseline 사용 |
| **Test5** | **0.944** | 14,645s | 600s | **644s** | high HI, EOL 모델 강하게 사용 |
| Test6 | 0.410 | 11,641s | 881s | 11,641s | HI는 낮지만 RMS/energy 위험. `13_EOLHazardGate`에서 재검증 필요 |

## 제출 후보 파일

| 우선순위 | 파일 | 새 이름 기준 전략 | 비고 |
|----------|------|------------------|------|
| 1순위 비교 후보 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_balanced_submission.xlsx` | `16_ScoreAware_CalibratedEnsemble` balanced | Test6 6000s clamp |
| 안정 백업 | `artifacts/results/05_HIBlend_Baseline_ChannelSym/submission_v24_v17v22_combined.xlsx` | `5_HIBlend_Baseline_ChannelSym` combined | 기존 best, Test6 clamp 없음 |
| 보수 후보 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_safe_submission.xlsx` | `16_ScoreAware_CalibratedEnsemble` safe | Test4/Test6 보수 |
| 위험 후보 | `artifacts/results/09_HIBlend_Baseline_DomainAdv/submission_v9_v17v26_*.xlsx` | `9_HIBlend_Baseline_DomainAdv` | full은 좋지만 Test5 high-HI에서 12,489s라 gate 전 단독 제출 금지 |

## 파일 경로 매핑

| 새 이름 | 주요 코드 | 모델 폴더 | 결과 파일 예시 |
|---------|----------|----------|----------------|
| 1_Baseline_1ch_TFTBiLSTM_GPR | `experiments/01_Baseline_1ch_TFTBiLSTM_GPR/pipeline.py` | `artifacts/models/01_Baseline_1ch_TFTBiLSTM_GPR/` | `artifacts/results/01_Baseline_1ch_TFTBiLSTM_GPR/lobo_results.csv` |
| 2_EOLDirect_4ch_WeightedRUL | `experiments/02_EOLDirect_4ch_WeightedRUL/pipeline.py` | `artifacts/models/02_EOLDirect_4ch_WeightedRUL/` | `artifacts/results/02_EOLDirect_4ch_WeightedRUL/lobo_v18.csv` |
| 3_HIBlend_Baseline_EOLDirect | `experiments/03_HIBlend_Baseline_EOLDirect/blend.py` | `artifacts/models/01...`, `artifacts/models/02...` | `artifacts/results/03_HIBlend_Baseline_EOLDirect/submission_v19_blend_*.xlsx` |
| 4_ChannelSym_EOLWeighted | `experiments/04_ChannelSym_EOLWeighted/pipeline.py` | `artifacts/models/04_ChannelSym_EOLWeighted/` | `artifacts/results/04_ChannelSym_EOLWeighted/lobo_v22.csv` |
| 5_HIBlend_Baseline_ChannelSym | `experiments/05_HIBlend_Baseline_ChannelSym/blend.py` | `artifacts/models/01...`, `artifacts/models/04...` | `artifacts/results/05_HIBlend_Baseline_ChannelSym/submission_v24_v17v22_*.xlsx` |
| 6_Dynamics_DTW_TFTBiLSTM | `experiments/06_Dynamics_DTW_TFTBiLSTM/pipeline.py` | `artifacts/models/06_Dynamics_DTW_TFTBiLSTM/` | `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/lobo_v25.csv` |
| 7_DomainAdv_Dynamics_TFT | `experiments/07_DomainAdv_Dynamics_TFT/pipeline.py` | `artifacts/models/07_DomainAdv_Dynamics_TFT/` | `artifacts/results/07_DomainAdv_Dynamics_TFT/lobo_v26.csv` |
| 8_HIBlend_Baseline_Dynamics | `experiments/08_HIBlend_Baseline_Dynamics/blend.py` | `artifacts/models/01...`, `artifacts/models/06...` | `artifacts/results/08_HIBlend_Baseline_Dynamics/submission_v8_v17v25_*.xlsx` |
| 9_HIBlend_Baseline_DomainAdv | `experiments/09_HIBlend_Baseline_DomainAdv/blend.py` | `artifacts/models/01...`, `artifacts/models/07...` | `artifacts/results/09_HIBlend_Baseline_DomainAdv/submission_v9_v17v26_*.xlsx` |
