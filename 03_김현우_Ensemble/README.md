# 03 김현우 Ensemble

루트는 실행/문서/아티팩트만 보이도록 정리했다. 공식 실험명은 `번호_방법명`을 사용하고, `v17`, `v22`, `v25` 같은 legacy ID는 추적용으로만 남긴다.

## Root Layout

```text
03_김현우_Ensemble/
├── README.md
├── VERSION_MAP.md
├── data/
├── docs/
├── experiments/
├── artifacts/
├── legacy/
├── run/
└── tools/
```

## Experiments

| 순서 | 공식 이름 | Legacy | 코드 위치 | 결과 위치 |
|------|-----------|--------|-----------|-----------|
| 1 | `1_Baseline_1ch_TFTBiLSTM_GPR` | v17 | `experiments/01_Baseline_1ch_TFTBiLSTM_GPR/` | `artifacts/results/01_Baseline_1ch_TFTBiLSTM_GPR/` |
| 2 | `2_EOLDirect_4ch_WeightedRUL` | v18 | `experiments/02_EOLDirect_4ch_WeightedRUL/` | `artifacts/results/02_EOLDirect_4ch_WeightedRUL/` |
| 3 | `3_HIBlend_Baseline_EOLDirect` | v19 | `experiments/03_HIBlend_Baseline_EOLDirect/` | `artifacts/results/03_HIBlend_Baseline_EOLDirect/` |
| 4 | `4_ChannelSym_EOLWeighted` | v22 | `experiments/04_ChannelSym_EOLWeighted/` | `artifacts/results/04_ChannelSym_EOLWeighted/` |
| 5 | `5_HIBlend_Baseline_ChannelSym` | v24 | `experiments/05_HIBlend_Baseline_ChannelSym/` | `artifacts/results/05_HIBlend_Baseline_ChannelSym/` |
| 6 | `6_Dynamics_DTW_TFTBiLSTM` | v25 | `experiments/06_Dynamics_DTW_TFTBiLSTM/` | `artifacts/results/06_Dynamics_DTW_TFTBiLSTM/` |
| 7 | `7_DomainAdv_Dynamics_TFT` | v26 | `experiments/07_DomainAdv_Dynamics_TFT/` | `artifacts/results/07_DomainAdv_Dynamics_TFT/` |
| 8 | `8_HIBlend_Baseline_Dynamics` | blend | `experiments/08_HIBlend_Baseline_Dynamics/` | `artifacts/results/08_HIBlend_Baseline_Dynamics/` |
| 9 | `9_HIBlend_Baseline_DomainAdv` | blend | `experiments/09_HIBlend_Baseline_DomainAdv/` | `artifacts/results/09_HIBlend_Baseline_DomainAdv/` |
| 13 | `13_EOLHazardGate_Calibrator` | new | `experiments/13_EOLHazardGate_Calibrator/` | `artifacts/results/13_EOLHazardGate_Calibrator/` |
| 14 | `14_RPMAwareOrderFeatures` | new | `experiments/14_RPMAwareOrderFeatures/` | `artifacts/results/14_RPMAwareOrderFeatures/` |
| 15 | `15_TrajectoryKNN_DTW_RUL` | new | `experiments/15_TrajectoryKNN_DTW_RUL/` | `artifacts/results/15_TrajectoryKNN_DTW_RUL/` |
| 16 | `16_ScoreAware_CalibratedEnsemble` | new | `experiments/16_ScoreAware_CalibratedEnsemble/` | `artifacts/results/16_ScoreAware_CalibratedEnsemble/` |

## Artifacts

| 폴더 | 내용 |
|------|------|
| `artifacts/models/01_Baseline_1ch_TFTBiLSTM_GPR/` | baseline 모델 |
| `artifacts/models/02_EOLDirect_4ch_WeightedRUL/` | EOLDirect 모델 |
| `artifacts/models/04_ChannelSym_EOLWeighted/` | ChannelSym 모델 |
| `artifacts/models/06_Dynamics_DTW_TFTBiLSTM/` | Dynamics 모델 |
| `artifacts/models/07_DomainAdv_Dynamics_TFT/` | DomainAdv 모델 |
| `artifacts/results/*/` | 실험별 점수, 플롯, 제출 파일 |
| `artifacts/logs/*/` | 실행 로그 |

## Legacy

폐기 실험은 `legacy/archive_failed/`에 보관한다.

| 공식 이름 | 위치 |
|-----------|------|
| `10_EOLWeightAblation_LowerWeights` | `legacy/archive_failed/pipelines/pipeline_v20.py` |
| `11_ThreeWayBlend_Baseline_EOLDirect_Ablation` | `legacy/archive_failed/blends/blend_v17_v18_v20.py` |
| `12_ThreeWayBlend_Baseline_EOLDirect_ChannelSym` | `legacy/archive_failed/blends/blend_v17_v18_v22.py` |

## Run Wrappers

검증 완료된 신규 실험은 `run/` wrapper로 실행한다.

```bash
bash run/run_13_eol_hazard_gate.sh
bash run/run_14_rpm_order_features.sh
bash run/run_15_trajectory_knn_dtw.sh
bash run/run_16_scoreaware_ensemble.sh
```

`13`, `15`, `16`은 새 구조에서 재실행 검증 완료. `14`는 원진동 전체 재처리라 시간이 오래 걸리지만 wrapper와 경로 보정은 완료되어 있다.

## Latest Submission Candidates

| 후보 | 파일 | 성향 |
|------|------|------|
| 추천 비교 후보 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_balanced_submission.xlsx` | `5_HIBlend` 기반, Test6 6000s clamp |
| 보수 후보 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_safe_submission.xlsx` | Test4 8400s, Test6 6000s, Test5 644s |
| 공격 후보 | `artifacts/results/16_ScoreAware_CalibratedEnsemble/16_scoreaware_aggressive_submission.xlsx` | pass 구간은 `9_DomainAdvBlend`, risk 구간은 safe clamp |
| 기존 안정 후보 | `artifacts/results/05_HIBlend_Baseline_ChannelSym/submission_v24_v17v22_combined.xlsx` | 기존 best, Test6 clamp 없음 |

## Path Utility

새 스크립트는 `tools/paths.py`를 사용한다.

```python
from tools.paths import RESULT_ROOT, MODEL_ROOT, result_dir, add_repo_to_path
add_repo_to_path()
```
