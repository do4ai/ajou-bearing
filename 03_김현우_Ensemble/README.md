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
| 16 | `16_ScoreAware_CalibratedEnsemble` | new (재설계: 임의 clamp 제거) | `experiments/16_ScoreAware_CalibratedEnsemble/` | `artifacts/results/16_ScoreAware_CalibratedEnsemble/` |
| **17** | `17_AsymOptimal_TrainBased` | **new (train-based KNN + 비대칭 페널티 직접 최적화)** | `experiments/17_AsymOptimal_TrainBased/` | `artifacts/results/17_AsymOptimal_TrainBased/` |
| **18** | `18_PerBearing_Robust` | **new (per-bearing best candidate selection, sens 0.488)** | `experiments/17_AsymOptimal_TrainBased/per_bearing_robust.py` | `artifacts/results/17_AsymOptimal_TrainBased/18_per_bearing_robust_*.{csv,xlsx}` |
| **19** | `19_EOLProgression_Robust` | **new (Train HI 곡선 fit + EOL bound cap, LOBO 0.75)** | `experiments/17_AsymOptimal_TrainBased/eol_progression_robust.py` | `artifacts/results/17_AsymOptimal_TrainBased/19_eol_progression_robust_*.csv` |
| **20** | `20_Consensus` | **new (train-based candidate 합의)** | `experiments/17_AsymOptimal_TrainBased/consensus.py` | `artifacts/results/17_AsymOptimal_TrainBased/20_consensus*.{csv,xlsx}` |
| **21** | `21_SubmissionMatrix` | **new (LOBO vs Sensitivity 종합)** | `experiments/17_AsymOptimal_TrainBased/submission_matrix.py` | `artifacts/results/17_AsymOptimal_TrainBased/21_submission*.csv` |
| **28** | `28_EOLRegressor_Specialist` | **new (Train rul_s≤15000 GBM+RF+ET 앙상블)** | `experiments/28_EOLRegressor_Specialist/` | `artifacts/results/28_EOLRegressor_Specialist/` |

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
bash run/run_05_hiblend_anchor.sh
bash run/run_final_anchor_then_26.sh
bash run/run_27_eol_classifier.sh
bash run/run_final_anchor_26_27.sh
```

`13`, `15`, `16`은 새 구조에서 재실행 검증 완료. `14`는 원진동 전체 재처리라 시간이 오래 걸리지만 wrapper와 경로 보정은 완료되어 있다.

`05 → 26` final-ready flow도 새 구조에서 재실행 검증 완료.

최종 테스트 베어링 이름을 직접 지정하려면:

```bash
TARGET_NAMES=Final1,Final2,Final3 bash run/run_final_anchor_then_26.sh
TARGET_NAMES=Final1,Final2,Final3 bash run/run_final_anchor_26_27.sh
```

## Latest Submission Candidates (v26 Train-Based)

> 임의 clamp 폐기. 모든 출력은 Train data로 학습된 회귀값. 600s 물리 하한만 허용.
> 자세한 의사결정: `artifacts/submissions/SUBMISSION_README.md`.

| 우선순위 | 파일 | 전략 | Sensitivity Mean | LOBO Score |
|---------|------|------|------------------|------------|
| **1순위** | `artifacts/submissions/팀이름_validation_1순위.xlsx` | `18_PerBearing_Robust` (베어링별 best mix) | **0.488** | — |
| **백업1** | `artifacts/submissions/팀이름_validation_백업1.xlsx` | `5_HIBlend_combined` (LOBO 검증 default) | 0.399 | **0.712** |
| **백업2** | `artifacts/submissions/팀이름_validation_백업2.xlsx` | `19_EOLProgression_Robust` (HI 곡선 fit) | 0.429 | **0.750** |

### 1순위 베어링별 선택

| Bearing | HI | 선택 모델 | 예측 (s) |
|---------|----|-----------|---------|
| Test1 | 0.46 | 28_eol_cons | 10,067 |
| Test2 | 0.50 | 28_eol_med | 10,998 |
| Test3 | 0.16 | 17_hybrid | 48,900 |
| Test4 | 0.45 | 28_eol_med | 9,545 |
| **Test5** | **0.94** | **5_HIBlend** | **644** |
| Test6 | 0.41 | 28_eol_med | 10,275 |

### 보고서·발표

| 산출물 | 파일 |
|--------|------|
| Report PDF (A4 1페이지) | `artifacts/results/17_AsymOptimal_TrainBased/팀이름_report.pdf` |
| PPT addendum (12 슬라이드) | `~/sensspace/projects/아주대/4_outputs/KSPHM_KIMM_RUL_v26_방법론_설명_HUFS_addendum.pptx` |
| Sensitivity heatmap | `artifacts/results/17_AsymOptimal_TrainBased/sensitivity_heatmap.png` |
| Per-bearing comparison | `artifacts/results/17_AsymOptimal_TrainBased/per_bearing_comparison.png` |

최종 테스트 제출 권장 flow:

```bash
TARGET_NAMES=Final1,Final2,Final3 bash run/run_final_anchor_26_27.sh
```

Final test용 후보는 public Validation 후보와 분리한다. 최종 테스트셋 공개 후 `26_FinalRobust_LOBOFrozenSelector`에서 Train1~4 LOBO 기준으로 freeze된 규칙만 적용한다.

과적합 방지 기준:

- 상세 문서: `docs/FINAL_ANTI_OVERFIT_PROTOCOL.md`
- public Validation/Test1~6에 맞춘 hand tuning은 final 후보에서 제외
- final test에는 fixed feature, fixed threshold, fixed KNN/window만 적용

## Path Utility

새 스크립트는 `tools/paths.py`를 사용한다.

```python
from tools.paths import RESULT_ROOT, MODEL_ROOT, result_dir, add_repo_to_path
add_repo_to_path()
```
