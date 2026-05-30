# 창의성 포지셔닝 — 선행연구 대비 차별점 (발표 창의성 10% 근거)

> 2024–2025 베어링 RUL 문헌 검토 후 정직한 차별화 서술. 과장 금지.

## 현재 주류 연구 (2024–2025)
- **아키텍처 중심**: BiLSTM 최적화, CBAM-CNN-LSTM, TCN-Transformer, LLM 전이학습 등.
- **불확실성 정량화**: LSTM + dropout/kernel density → 점추정 + 확률분포 동시 산출.
- 대부분 **대칭 손실(RMSE/MAE)** 최적화 후, RUL을 점추정으로 직접 회귀.

출처:
- [Optimized BiLSTM, Sensors 2025](https://www.mdpi.com/1424-8220/25/14/4351)
- [LSTM Uncertainty Quantification, PMC9228128](https://pmc.ncbi.nlm.nih.gov/articles/PMC9228128/)
- [CBAM-CNN-LSTM, PMC11768707](https://pmc.ncbi.nlm.nih.gov/articles/PMC11768707/)
- [LLM transfer RUL, arXiv 2501.07191](https://arxiv.org/pdf/2501.07191)

## 우리 접근의 차별점 (정직한 서술)

### 1. 비대칭 점수 직접 최적화 점추정 (argmax_p E_R[A(p,R)])
- 주류는 RMSE 최적화 후 점추정을 그대로 제출 → 평가식의 비대칭성(늦은 예측 2.5×)과 불일치.
- 우리는 train RUL 분포 P(R|state)에서 **평가 점수 A의 기대값을 최대화하는 단일 점**을 선택.
- 즉 모델 출력이 아니라 **평가 metric 자체를 objective로 한 의사결정 이론적 점추정**.
- 관련: uncertainty quantification 연구가 분포를 산출하지만, 그 분포로 비대칭-최적 점을 고르는 단계는 드묾.

### 2. Per-Bearing HI-band Regime Selection
- 주류는 전 베어링에 단일 global 모델.
- 우리는 베어링별 열화 단계(HI-band)에 따라 train-based 후보 중 비대칭 기대값 최대를 선택.
- LOBO 검증으로 generalize 입증(0.519 > naive 0.460) — 단순 model averaging 대비 robust.

### 3. Train-Based Only 원칙 + 임의 clamp 폐기
- EOL 폭주 방지를 위해 임의 안전치(6000s 등)를 박는 대신, train RUL 분포가 자연 상한을 제공.
- 모든 출력이 train data 학습 결과 → 챌린지 정신(데이터로 예측)에 완전 정합.

### 4. HI-Velocity Anomaly Detection (Test5)
- 동일 관측 윈도우(8.17h)에서 train HI≈0.5인데 Test5만 0.944 → train 분포 밖 outlier.
- 절대 HI level이 아닌 **동일 시간 상대 위치**로 비정상 열화 검출 → 짧은 RUL 의사결정 근거.

### 5. Conformal 예측구간 (LOBO 잔차 기반, 점추정 보완)
- 주류 문헌의 불확실성 정량화와 달리, 우리는 **점추정을 메인**으로 하되 LOBO 16점 잔차의
  percentage-error 경험분포로 **calibrated 구간을 ALSO 제공** (split-conformal 유사, train-only).
- **핵심 발견**: LOBO median Er = **+23.4% (구조적 under-predict)** → 선택법이 자연히 보수(early)
  방향 → 늦은 예측 2.5× 페널티에 정합. 이는 asym-optimal 점추정의 자연 귀결 (우연 아님).
- 구간 폭이 큰 것은 train 4 베어링 한계의 **정직한 반영** (과대 정밀 주장 회피 = 합리성).
- Test5는 90% 상한도 0.56h로 짧음 → 짧은 RUL 의사결정이 불확실성 하에서도 robust.
- 산출: 27_conformal_intervals.csv (점추정은 제출본 불변 — 순수 additive).

### 6. 물리 기반 열화율 RUL + 2축 건강평가 (블랙박스 대비 해석가능)
- 주류 문헌은 BiLSTM/CBAM/TCN 등 **블랙박스 회귀**로 RUL을 직접 추정.
- 우리는 데이터로 학습한 HI(건강지표)에 **물리 모델**을 입혀 해석가능한 RUL을 추가 도출:
  `RUL = elapsed × (1 − HI) / HI` (HI를 고장 HI≈1.0까지의 일정 평균 열화율로 외삽; 파라미터 거의 無).
  - **검증**: train progression LOBO asym **0.586**, severity-gate 결합 시 **0.592** — 단순 train 방법 중 최고
    (per-bearing 선택 0.519 / HI-band prior 0.508 대비). **단일 일관 규칙**(per-bearing 튜닝 아님)으로 달성.
  - **단, n=4 부트스트랩(36)으로 정직 경계**: 95% CI 전부 중첩(physics-gated [0.49,0.69] 등), P(gated>avg-rate)=0.42 → 우열은 **통계적으로 비결정적**. "물리 방법군이 기존 트랙과 동급~약간 상위, n=4라 결정적 아님"으로 서술 (과대 정밀 주장 회피 = 합리성).
- **2축 건강평가** = 진행도(HI) × 심각도(energy_ratio·rms_multi). 임계는 train near-EOL p90(데이터 유래).
  - 이 2축이 두 anomaly를 **하나의 원리로 통합**: Test5(HI 0.94+energy>EOL)와 Test6(HI 0.41이지만
    energy 23.3=train EOL의 2배 → **숨은 급성 열화**)를 동일 프레임으로 설명 → 짧은 RUL 결정의 물리 근거.
- **정직성**: 곡률 보정(p>1)·국소 기울기 변형은 LOBO에서 이득 없어 **기각**(단순 모델 유지), HI-단독
  모델이 Test6를 과대예측한 것도 energy 축으로 **자체정정** → "복잡도보다 검증된 단순성" 원칙 일관.
- 산출: experiments/32_DegradationRate_RUL/ (predict/convex/severity/physics_gated, 전부 train-only).

## 발표 시 정직한 표현 (권장 워딩)
- "우리는 새 아키텍처를 제안하기보다, **평가식 비대칭성을 점추정 단계까지 일관되게 반영**한 의사결정 파이프라인을 설계했다."
- "선행 불확실성 정량화 연구와 달리, 산출된 분포에서 **비대칭 기대점수 최적점**을 선택한다."
- 과장 금지: "최초"라는 표현 대신 "흔치 않은 조합", "평가 정합적 설계"로 서술.
- "블랙박스 회귀에 더해, **데이터 학습 HI에 물리 열화 모델**을 입혀 해석가능한 RUL을 교차검증했고(LOBO 0.59), HI(진행도)×energy(심각도) **2축 건강평가**로 Test5·Test6 이상치를 하나의 원리로 설명한다."
