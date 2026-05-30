# Final Test Anti-Overfitting Protocol

## 배경

운영위 공지에 따르면 레이블링되지 않은 최종 테스트셋이 별도로 공개될 예정이다. 따라서 지금까지 공개된 Train/Validation에 과도하게 맞춘 모델은 Final score에서 무너질 수 있다.

최종 제출 원칙은 다음과 같다.

- 학습과 하이퍼파라미터 선택은 Train1~4 LOBO 기준으로 고정한다.
- 현재 공개된 Validation/Test1~6의 개별 이름이나 개별 결과에 맞춘 수동 보정은 최종 테스트에 직접 사용하지 않는다.
- 최종 테스트셋에는 label이 없으므로, feature 추출과 inference만 수행한다.
- transductive domain adaptation을 쓰더라도 label이나 leaderboard feedback 없이 사전에 고정된 규칙만 적용한다.

## 현재 실험의 과적합 위험 분류

| 실험 | 과적합 위험 | 최종 테스트 사용 정책 |
|---|---|---|
| `1_Baseline_1ch_TFTBiLSTM_GPR` | 낮음 | 사용 가능. LOBO 기반 baseline |
| `2_EOLDirect_4ch_WeightedRUL` | 낮음~중간 | 사용 가능. EOL 가중치가 강해 last 편향 주의 |
| `3_HIBlend_Baseline_EOLDirect` | 중간 | threshold가 LOBO grid 기반이면 사용 가능 |
| `4_ChannelSym_EOLWeighted` | 낮음~중간 | 사용 가능. EOL branch |
| `5_HIBlend_Baseline_ChannelSym` | 중간 | 기존 안정 anchor로 사용 가능 |
| `6_Dynamics_DTW_TFTBiLSTM` | 중간~높음 | Train3 폭주 때문에 단독 사용 금지 |
| `7_DomainAdv_Dynamics_TFT` | 높음 | Train3/Test5 폭주 때문에 단독 사용 금지 |
| `13_EOLHazardGate_Calibrator` | 중간 | 규칙을 Train-only 기준으로 freeze하면 사용 가능 |
| `14_RPMAwareOrderFeatures` | 낮음 | 물리 피처라 사용 가능 |
| `15_TrajectoryKNN_DTW_RUL` | 중간 | 최종 테스트 개별 튜닝 없이 fixed k/feature/window로만 사용 가능 |
| `16_ScoreAware_CalibratedEnsemble` | 중간~높음 | 현재 Test1~6 decision이 섞여 있으므로 final용 별도 frozen 버전 필요 |
| `17_ConditionalMADA_StageAlignment` | 중간 | 진단용. source weighting만 고정 규칙이면 사용 가능 |
| `18_MMD_CORAL_SourceWeighting` | 중간 | label-free지만 target 분포 의존. fixed rule만 허용 |
| `19_DomainSpecificResidual_Calibrator` | 높음 | final 후보 아님 |
| `20_UncertaintyWeighted_TargetAdaptation` | 중간 | fixed uncertainty rule이면 사용 가능 |
| `21_VMD_CBAM_FeatureDenoising` | 낮음~중간 | feature diagnostic으로 사용 가능, 단독 제출 부적합 |
| `22_ParallelTCNTransformer_Branch` | 높음 | LOBO last 불안정. 단독 사용 금지 |
| `23_FullConditionalMADA_Trainable` | 높음 | 과대예측. 단독 사용 금지 |
| `24_Wasserstein_SourceWeighted_RUL` | 높음 | EOL 과대예측. 단독 사용 금지 |
| `25_StageAwareTransformer_DA` | 중간~높음 | EOL 진단 보조. 단독 사용 금지 |

## 최종 제출 후보 분리

### Public Validation 후보

현재 공개 Validation/Test1~6에서 비교하기 위한 후보:

- `16_scoreaware_balanced_submission.xlsx`
- `16_scoreaware_safe_submission.xlsx`
- `20_uncertainty_weighted_submission.xlsx`

이 후보들은 현재 Test1~6의 위험 패턴을 반영했으므로, 최종 테스트에 그대로 가져가면 과적합 가능성이 있다.

### Final Test 후보

최종 테스트용 후보는 다음 기준으로 새로 만든다.

- anchor: `5_HIBlend_Baseline_ChannelSym`
- EOL gate: Train1~4 LOBO에서 고정된 threshold만 사용
- dynamics/order/trajectory: fixed feature set, fixed window, fixed k
- no per-bearing manual correction
- no leaderboard-driven threshold update

권장 이름:

- `26_FinalRobust_LOBOFrozenSelector`

## 최종 테스트 실행 절차

1. 최종 Test 데이터를 `data/raw/Test*` 또는 별도 final 이름으로 변환한다.
2. 기존 Train1~4로 fit된 모델/스케일러만 사용한다.
3. 최종 Test label은 절대 사용하지 않는다.
4. `26_FinalRobust_LOBOFrozenSelector`로 inference한다.
5. public validation 성능이 아니라 LOBO worst-case와 late-risk를 기준으로 최종 후보를 고른다.

## Final Test Ready 실행 명령

현재 구조에서는 `05` anchor 생성 후 `26` selector를 이어서 실행한다.

```bash
bash 03_김현우_Ensemble/run/run_final_anchor_then_26.sh
```

최종 테스트 베어링 이름이 `Test1~Test6`가 아니라면 `TARGET_NAMES` 환경변수로 지정한다.

```bash
TARGET_NAMES=Final1,Final2,Final3 bash 03_김현우_Ensemble/run/run_final_anchor_then_26.sh
```

산출물:

- anchor: `artifacts/results/05_HIBlend_Baseline_ChannelSym/submission_v24_v17v22_debug.csv`
- final-safe: `artifacts/results/26_FinalRobust_LOBOFrozenSelector/26_final_robust_submission.xlsx`
- EOL classifier enhanced: `artifacts/results/27_EOLClassifier_LOBOCalibrated/27_eol_classifier_submission.xlsx`

주의:

- `05` anchor가 없으면 `26`은 train-only fallback으로 동작하지만, fallback 단독은 LOBO last가 약하므로 권장하지 않는다.
- 최종 제출은 반드시 `05 anchor + 26 selector` 순서로 생성한다.

## 27 EOL Classifier 정책

`27_EOLClassifier_LOBOCalibrated`는 Train1~4의 `rul_s`에서 파생한 label만 사용한다.

```text
EOL_1200 = rul_s <= 1200
EOL_2400 = rul_s <= 2400
EOL_3600 = rul_s <= 3600
EOL_6000 = rul_s <= 6000
```

목적:

- RUL 회귀가 놓치는 EOL 위험을 별도 classifier로 감지.
- final test에서 late-prediction 폭주를 줄임.
- public Validation/Test1~6의 수동 label은 사용하지 않음.

현재 LOBO 평균:

| label | recall | positive_rate | AUC | AP |
|---|---:|---:|---:|---:|
| EOL_1200 | 0.375 | 0.008 | 0.966 | 0.556 |
| EOL_2400 | 0.750 | 0.048 | 0.973 | 0.684 |
| EOL_3600 | 0.708 | 0.087 | 0.964 | 0.686 |
| EOL_6000 | 0.875 | 0.174 | 0.971 | 0.756 |

사용 판단:

- `27`은 `26` 위에 classifier probability를 추가한 final-safe 후보.
- Public validation에서는 `26`과 동일하게 Test6 3000s, Test5 644s.
- Final test에서는 EOL_6000 recall 보강 효과가 기대된다.

실행:

```bash
bash 03_김현우_Ensemble/run/run_27_eol_classifier.sh
```

최종 통합 flow:

```bash
TARGET_NAMES=Final1,Final2,Final3 bash 03_김현우_Ensemble/run/run_final_anchor_26_27.sh
```

## 모델 선택 기준

최종 선택 점수는 단순 평균이 아니라 보수적 기준을 사용한다.

```text
robust_score = 0.45 * last_mean
             + 0.25 * worst_last
             + 0.20 * full_mean
             + 0.10 * eol_recall
```

폐기 조건:

- LOBO 중 하나라도 마지막 RUL 600s에서 3600s 초과 예측이 반복되면 폐기.
- Train3 last 폭주가 있으면 EOL branch로 사용 금지.
- Test feature를 보고 hand-tuned threshold를 바꾸면 final 후보에서 제외.

## 발표 대응 포인트

상위권 발표 가능성을 고려해 다음을 강조한다.

- random split이 아니라 LOBO로 베어링 단위 일반화 검증.
- public validation에 맞춘 hand tuning을 final test용 모델에서 분리.
- 공식 점수의 late prediction penalty 때문에 EOL risk를 보수적으로 관리.
- 물리 기반 order feature와 channel symmetry로 설명 가능성 확보.
- 복잡한 DA/MADA도 시도했지만, 작은 데이터에서는 gate/calibration이 더 robust했다는 ablation 근거 제시.
