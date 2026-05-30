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

---

# v26 Train-Based Update (2026-05-29, 임의 clamp 폐기 후)

> **핵심 전환**: 기존 16_ScoreAware는 EOL gate에서 2400/3600/6000 임의 clamp를 사용 → 챌린지 정신 위반.
> v26부터 **모든 출력은 train data로 학습된 회귀값**. 600s 물리 하한만 허용. 위 "제출 후보" 표는 deprecated.

## 신규 train-based 메서드

| 메서드 | 방법 | LOBO | Sensitivity mean |
|--------|------|------|------------------|
| 17_AsymOptimal | HI-state KNN(K=20) → train RUL 분포 → argmax_p E[A(p,r)] | full 0.520 | 0.401 |
| 18_PerBearing_Robust | 9 candidate × HI-band grid → 베어링별 best | — | **0.488** |
| 19_EOLProgression_Robust | HI 곡선 fit + EOL bound cap → progression 역산 | **0.750** | 0.429 |
| 20_Consensus | 7 candidate median/asym/trimmed/weighted | — | 0.458 |
| 21_SubmissionMatrix | LOBO vs Sensitivity 통합 (anti-correlated 발견) | — | — |
| 22_HIVelocity | HI@8.17h → Test5 anomaly 발견 | 0.568 | — |
| 23_BetaSweep | conservative tilt β=0.95 worst-case 개선 | — | worst 0.40→0.42 |
| 28_EOLRegressor_Specialist | train rul_s≤15000 GBM+RF+ET 앙상블 | last 0.054 | mid-life robust |

## v26 제출 후보 (현행, 모두 train-based)

| 우선순위 | 파일 | 전략 | Sens / LOBO |
|----------|------|------|-------------|
| 1순위 | `artifacts/submissions/HUFS_validation_1순위.xlsx` | 18_PerBearing_Robust | Sens 0.488 |
| 1순위-cons | `artifacts/submissions/HUFS_validation_1순위_conservative.xlsx` | β=0.95 worst-robust | worst 0.423 |
| 백업1 | `artifacts/submissions/HUFS_validation_백업1.xlsx` | 5_HIBlend | LOBO 0.712 |
| 백업2 | `artifacts/submissions/HUFS_validation_백업2.xlsx` | 19_EOLProgression | LOBO 0.750 |

자세한 의사결정: `artifacts/submissions/FINAL_DECISION.md`.

## 핵심 통찰 (v26)

1. **Test5 anomaly**: 동일 8.17h 관측에서 train은 모두 HI~0.5, Test5만 0.944 → train 분포 밖 비정상 빠른 열화 → 짧은 RUL(644s) 정당.
2. **LOBO ↔ Sensitivity anti-correlation**: LOBO는 train 마지막 600s 라벨에 fit, Sensitivity는 HI-band prior 가정 → 두 관점 모두 robust한 후보가 안전.
3. **Test3** (HI=0.16): 정상보다 느린 열화 → 긴 RUL(48900s).
4. **위험 분산**: 1순위(sensitivity prior) / 백업1(LOBO train-fit) / 백업2(EOL physics) 세 독립 가설.

## v27 물리 열화 방법론 (experiments/32_DegradationRate_RUL/) — 신규, train-only

주류 RUL 문헌(블랙박스 DL) 대비 **물리·해석가능** 차별점. HI에 물리 모델을 입혀 RUL 교차검증.

| 메서드 | 핵심 | LOBO progression asym |
|--------|------|----------------------|
| 32 DegradationRate (`predict.py`) | `RUL=elapsed×(1−HI)/HI` (평균 열화율, 고장 HI≈1.0) | 0.586 |
| 33 Convex (`convex_calibrated.py`) | `HI(t)=(t/T)^p` 곡률 보정, LOBO로 p 캘리브 | 0.550 (p≈1 최적 → 이득 無, 단순모델 재확정) |
| 34 SeverityTwoAxis (`severity_two_axis.py`) | 진행도(HI) × 심각도(energy/rms), 임계=train EOL p90 | (분류) Test5/6 severe, Test1~4 정상 |
| 35 PhysicsGated (`physics_gated.py`) | severity-gate + avg-rate 단일규칙 | 0.592 (EOL 버킷 거침) |
| (비교) per-bearing 선택 / HI-prior | 기존 트랙 | 0.519 / 0.508 |

**⚠️ n=4 부트스트랩 (36_bootstrap_lobo_ci.py)**: 4 베어링 복원추출 256표본 95% CI — HI-only [0.37,0.79]·avg-rate [0.52,0.70]·physics-gated [0.49,0.69] **전부 중첩**, P(physics-gated>avg-rate)=0.42. → **방법 간 우열은 n=4 한계로 통계적으로 비결정적**(점추정 "최고"는 표본노이즈 내). 정직 서술: "물리 방법군이 per-bearing(0.52)·HI-prior(0.51)와 **동급~약간 상위**, 결정적 아님".

**핵심 통찰 (v27)**:
- **2축 건강평가**가 Test5(HI 0.94+energy>EOL)와 Test6(HI 0.41이나 energy 23.3=train EOL 2배 → **숨은 급성 열화**)를 하나의 원리로 통합 → Test6 짧은 RUL 정당화.
- 독립 물리 모델들이 일관되게 **mid-life(Test1/2/4) 긴 RUL** 지지 → per-bearing 후보 mid-life 과소예측 가능성 (트랙 결정 입력). 상세: `docs/TRACK_RECONCILIATION.md` §9–10.
- 정직성: 곡률·국소-slope 변형은 LOBO 이득 없어 기각, HI-단독 Test6 오판은 energy 축으로 자체정정, 점추정 우열은 부트스트랩으로 비결정적 인정 → "검증된 단순성 + 정직한 불확실성".
