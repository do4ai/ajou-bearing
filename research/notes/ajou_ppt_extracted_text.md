## ppt/slides/slide1.xml

KSPHM-KIMM 2026 베어링 RUL 예측 방법론 v24: Fast Kurtogram · DTC-VAE HI · TCN-Transformer · BiLSTM · HI-conditioned Ensemble 기초 개념부터 제출 판단까지 설명 가능한 발표자료 RUL = Remaining Useful Life = 남은 수명(초) 핵심 메시지 HUFS · Navy Theme

## ppt/slides/slide10.xml

Feature v24 피처 엔지니어링 전체 10 KSPHM-KIMM 2026 Bearing RUL Prediction 시간 통계 Envelope/Order 4채널 대칭 운전/상태 주의: Validation/Test의 운전조건 CSV는 현재 비어 있어, 최종 예측에서는 진동 기반 피처와 HI가 더 중요하다.

## ppt/slides/slide11.xml

Dynamics 동역학 피처: 현재값보다 '변화 속도'가 중요 11 KSPHM-KIMM 2026 Bearing RUL Prediction 현재값 피처 동역학 피처 왜 중요한가? HI가 높아도 진짜 EOL인지, 한 번 튄 노이즈인지 구분해야 한다. 특히 Test5 판단에 중요하다. dynamic candidates: d_HI, slope_HI_5, acc_HI, roll_std_HI, d_env_kurt, d_energy_ratio

## ppt/slides/slide12.xml

Model DTC-VAE: 건강지수 HI를 만드는 모델 12 KSPHM-KIMM 2026 Bearing RUL Prediction VAE Variational AutoEncoder. 많은 피처를 작은 latent vector로 압축하고 다시 복원하도록 학습한다. DTC Deep/Degradation Temporal Consistency. 시간이 지날수록 건강지수가 대체로 증가하도록 제약을 건다. HI latent_0을 0~1로 정규화해 Health Indicator로 사용한다. 높을수록 고장에 가깝다고 해석한다. loss = reconstruction + KL + monotonicity + trendability

## ppt/slides/slide13.xml

Model CUSUM과 FPT: 열화 시작점 찾기 13 KSPHM-KIMM 2026 Bearing RUL Prediction CUSUM Cumulative Sum. 초기 정상 구간의 평균/표준편차를 기준으로, 이후 상태가 누적해서 벗어나는지 감지한다. FPT First Predicting Time. HI가 정상 범위를 벗어나 열화가 시작됐다고 판단하는 첫 시점이다. Piecewise RUL FPT 전에는 건강 구간으로 보고 RUL을 평탄하게 둔다. FPT 후에는 EOL까지 선형 감소시킨다.

## ppt/slides/slide14.xml

Model TCN과 Transformer 14 KSPHM-KIMM 2026 Bearing RUL Prediction TCN Temporal Convolutional Network. 시간 방향 1D convolution으로 최근 패턴과 국소 변화 흐름을 잡는다. Dilated convolution으로 더 넓은 과거도 본다. Transformer Self-attention으로 시점들 사이의 관계를 본다. 어떤 시점이 마지막 RUL 예측에 중요한지 학습한다. 우리 TFTModel Variable Selection → TCN → Transformer Encoder → RUL Head. 이름은 TFT지만 실제로 TCN-Transformer 하이브리드다. Feature sequence(10 steps) → VSN → TCN → Transformer → RUL

## ppt/slides/slide15.xml

Model TFT와 BiLSTM 15 KSPHM-KIMM 2026 Bearing RUL Prediction TFT는 무슨 줄임말? Temporal Fusion Transformer. 원래는 시계열 예측용 Transformer 계열 모델이다. 우리 코드는 TFT 원 논문 전체 구현은 아니지만, variable selection과 temporal attention 구조를 차용했다. BiLSTM은 왜 같이 쓰나? Bidirectional LSTM. 작은 데이터에서 Transformer보다 안정적으로 추세를 잡는 경우가 있어, 아키텍처 다양성을 확보하기 위해 함께 앙상블한다. 모델 강점 약점 TCN/TFT 국소 패턴+attention 작은 데이터 과적합 위험 BiLSTM 추세 안정성 급격 변화는 둔할 수 있음 Ensemble 상호 보완 복잡도 증가

## ppt/slides/slide16.xml

Training 학습: 증강, 가중치, 점수 기반 Loss 16 KSPHM-KIMM 2026 Bearing RUL Prediction 증강 EOL 가중치 v22는 RUL이 가장 작은 마지막 5개 샘플에 50배, 다음 15개 샘플에 10배 가중치를 준다. Loss 늦은 예측을 더 강하게 벌주는 asymmetric MSE와 공식 점수를 흉내 낸 score loss를 섞는다. loss = α·weighted_asym_MSE + (1-α)·score_loss

## ppt/slides/slide17.xml

Validation LOBO 검증: 베어링 하나를 통째로 숨긴다 17 KSPHM-KIMM 2026 Bearing RUL Prediction Train1을 숨기고 Train2~4로 학습한 뒤 Train1을 예측 Train2, Train3, Train4도 같은 방식으로 반복 랜덤 split은 같은 베어링의 앞/뒤가 섞여 실제 일반화 성능을 과대평가할 수 있음 Validation/Test는 새로운 베어링이므로 LOBO가 더 정직한 검증이다. Full 전체 RUL 곡선 평균 점수 Last 마지막 관측 시점만 평가 Combined 0.7×Full + 0.3×Last

## ppt/slides/slide18.xml

v24 v17과 v22의 역할 18 KSPHM-KIMM 2026 Bearing RUL Prediction v17: 안정적인 기본 모델 1채널 중심의 강건한 피처와 TFT/BiLSTM 앙상블. 중간 수명 구간과 일반적인 패턴에서 안정적이다. v22: EOL 민감 모델 4채널 대칭 피처, 온도 차이, EOL 50배 가중치, 늦은 예측 5배 페널티. 고장 직전 신호에 더 민감하다. v24 = HI가 낮으면 v17, HI가 높으면 v22

## ppt/slides/slide19.xml

Pipeline v17 상세: 안정적인 기본 파이프라인 19 KSPHM-KIMM 2026 Bearing RUL Prediction 단계 내용 발표 포인트 1. 입력 4채널 진동 중 ch0 중심 + 전체 RMS 보조 가장 안정적인 기준 채널을 중심으로 시작 2. 전처리 Fast Kurtogram → 최적 충격 대역 선택 → Band-pass 결함 충격이 잘 보이는 주파수 대역을 먼저 찾음 3. Envelope Hilbert envelope → envelope spectrum 충격 반복 패턴을 결함 order 에너지로 변환 4. 피처 RMS, kurtosis, crest, p2p, BPFI/BPFO/BSF/FTF energy, SNR 시간 통계 + 결함 주파수 피처 결합 5. HI DTC-VAE latent_0 → 0~1 Health Indicator 열화 방향성을 갖는 상태지수 생성 6. 라벨 CUSUM FPT + Piecewise RUL 초기 건강 구간과 열화 구간을 분리 v17은 전체적으로 안정적인 베이스라인이다. Test1·2·3·4·6에서는 최종 v24가 거의 v17 예측을 그대로 사용한다.

## ppt/slides/slide2.xml

Overview 발표 흐름 02 KSPHM-KIMM 2026 Bearing RUL Prediction 01  챌린지 문제 Train은 EOL까지, Validation은 EOL 전까지만 제공 02  기초 신호 개념 kurtosis, envelope, Hilbert, order, 600s 03  피처 엔지니어링 Fast Kurtogram, 4채널 피처, 동역학 피처 04  모델 구조 DTC-VAE HI, TCN-Transformer, BiLSTM 05  v24 앙상블 HI가 낮으면 v17, 높으면 v22 06  제출 판단 Test5 644s의 의미와 리스크

## ppt/slides/slide20.xml

Pipeline v22 상세: EOL 민감 파이프라인 20 KSPHM-KIMM 2026 Bearing RUL Prediction 단계 내용 v17 대비 변화 1. 입력 4채널 전체 진동 사용 ch0 중심에서 4채널 전체로 확장 2. 채널별 피처 각 채널의 RMS, kurtosis, envelope, order energy 채널별 결함 감도 차이를 반영 3. 채널 대칭 max/min/range/std/top2, correlation, energy ratio 특정 채널에서만 보이는 고장 대응 4. 운전/온도 rpm, torque, temp_diff, temp_max, temp_ratio 가능한 운전조건 정보를 함께 사용 5. HI 4채널 피처 기반 DTC-VAE HI v24 블렌딩 기준으로 사용 6. 라벨 rul_s 직접 학습 + 600s 하한 EOL 직전 예측에 더 직접적으로 맞춤 v22는 고장 직전 신호를 놓치지 않기 위해 설계했다. Test5에서 HI가 0.944로 높게 나타나 v22 예측이 강하게 반영된다.

## ppt/slides/slide21.xml

Training 데이터 증강과 불균형 보완 상세 21 KSPHM-KIMM 2026 Bearing RUL Prediction Gaussian Noise Mixup EOL Weight Score Loss 공식 A_RUL 점수를 직접 흉내 낸 loss를 섞는다. 늦은 예측을 줄이는 방향으로 학습한다. Curriculum 초반에는 MSE 중심으로 안정적으로 회귀를 배우고, 후반에는 score loss 비중을 높인다. 주의 증강은 Train fold 안에서만 적용한다. LOBO 검증 베어링과 Validation/Test에는 증강을 적용하지 않는다. loss = α·weighted_asym_MSE + (1-α)·score_loss

## ppt/slides/slide22.xml

Training 모델 학습부터 스코어 산출까지 22 KSPHM-KIMM 2026 Bearing RUL Prediction 순서 처리 결과물 1 Train1~4 중 하나를 검증 베어링으로 제외 LOBO fold 구성 2 나머지 Train 베어링으로 scaler fit 데이터 누수 방지 3 최근 10개 측정을 하나의 sequence로 구성 SEQ_LEN = 10 4 TFT 5 seeds + BiLSTM 5 seeds 학습 fold당 10개 모델 5 각 모델의 예측을 median/mean/trimmed로 결합 outlier seed 영향 축소 6 asym_score로 Full, Last 계산 공식 점수 기반 내부 검증 7 Full/Last/Combined 비교 블렌딩 파라미터 선택 8 Test1~6는 모든 fold 모델로 ensemble 최종 submission.xlsx 생성 핵심은 랜덤 split이 아니라 베어링 단위 LOBO 검증이다. 새로운 Validation 베어링을 맞히는 실제 문제와 가장 비슷한 평가 방식이다.

## ppt/slides/slide23.xml

v24 HI-conditioned Blending 23 KSPHM-KIMM 2026 Bearing RUL Prediction v24 combined 최적값 해석 HI < 0.8이면 거의 v17을 사용한다. HI > 0.8이면 거의 v22를 사용한다. 실제 효과 Test1,2,3,4,6은 v17. Test5만 HI가 0.94라 v22 쪽으로 급격히 전환된다.

## ppt/slides/slide24.xml

Score LOBO 검증 점수: Full · Last · Combined 24 KSPHM-KIMM 2026 Bearing RUL Prediction Full Train 베어링의 전체 RUL 곡선을 모든 측정 시점에서 평가한 평균 점수 Last 각 Train 베어링의 마지막 측정 시점만 평가한 점수. 공식 제출 상황과 가장 유사 Combined 후보 Full Last Combined 해석 v19 best full (v17+v18) 0.629 0.600 0.620 전체 곡선 안정 v19 best combined (v17+v18) 0.603 0.978 0.715 Last 매우 강함 v24 best full (v17+v22) 0.637 0.308 0.538 Full 최고, Last 약함 v24 best combined (v17+v22) 0.599 0.975 0.712 현재 1순위 제출 후보 v23 best combined (3-way) 0.581 1.000 0.706 Last는 최고, Full 낮음 점수는 공식 A_RUL 기반이며 1.0에 가까울수록 좋다. 공식 제출은 마지막 시점 RUL 1개에 가까우므로 Last를 중요하게 보되, Last 과적합을 막기 위해 Full도 함께 본다.

## ppt/slides/slide25.xml

Result 실험 결과 시각화: 후보별 LOBO 점수 25 KSPHM-KIMM 2026 Bearing RUL Prediction 0.00 0.25 0.50 0.75 1.00 0.629 0.600 0.620 0.603 0.978 0.715 0.637 0.308 0.538 0.599 0.975 0.712 0.581 1.000 0.706 Full Last Combined 해석 1 Full 최고는 v24 best full이지만 Last가 낮아 제출 후보로는 위험하다. 해석 2 v24 best combined는 Last를 크게 살리면서 Full도 유지한 후보다.

## ppt/slides/slide26.xml

Result 실험 결과 시각화: v24 LOBO 예측 곡선 26 KSPHM-KIMM 2026 Bearing RUL Prediction 읽는 법 검은 선은 실제 RUL, 파란/초록 선은 v17/v22, 빨간 선은 HI-blend 예측이다. LOBO 조건에서 예측 곡선을 비교한다.

## ppt/slides/slide27.xml

Result 실험 결과 시각화: Validation 최종 RUL 예측 27 KSPHM-KIMM 2026 Bearing RUL Prediction Test1 8.90h Test2 9.32h Test3 1.79h Test4 3.92h Test5 0.18h Test6 3.23h Bearing RUL seconds RUL hours Test1 32,035 8.90 Test2 33,556 9.32 Test3 6,449 1.79 Test4 14,113 3.92 Test5 644 0.18 Test6 11,641 3.23 핵심 시각화 포인트 Test5만 0.18h로 매우 짧다. v24는 Test5를 다음 10분 안팎 EOL로 본다.

## ppt/slides/slide28.xml

Result 실험 결과 시각화: Test5 민감도 28 KSPHM-KIMM 2026 Bearing RUL Prediction 0.00 0.25 0.50 0.75 1.00 0.774 0.000 0.000 600s 0.526 0.000 0.007 1,200s 0.337 0.000 0.972 3,000s 0.290 0.093 0.493 6,000s 0.273 0.960 0.376 10,000s 0.265 0.637 0.328 15,000s v24 644s v19 10,118s v24 full 2,938s 해석 Test5 실제 RUL이 6,000s 이하라면 v24가 유리하다. 리스크 Test5 실제 RUL이 10,000s 근처라면 v19가 유리하다.

## ppt/slides/slide29.xml

Result v24 최종 Validation 예측 29 KSPHM-KIMM 2026 Bearing RUL Prediction Bearing HI_last v17 v22 v24 final Test1 0.463 32,035s 39,775s 32,035s Test2 0.500 33,556s 39,303s 33,556s Test3 0.165 6,449s 7,311s 6,449s Test4 0.448 14,113s 9,818s 14,113s Test5 0.944 14,645s 600s 644s Test6 0.410 11,641s 881s 11,641s 핵심 v24는 대부분 v17을 유지하고, Test5만 고장 직전으로 판단한다. 리스크 Test5가 실제로 오래 남았다면 v19가 더 좋을 수 있다. 승부처는 Test5다.

## ppt/slides/slide3.xml

Challenge 정식 챌린지 정보 03 KSPHM-KIMM 2026 Bearing RUL Prediction 목표: Train 데이터로 모델을 학습해 Validation 베어링의 잔여수명(RUL)을 예측 Train: 열화시험 중단 조건(EOL)에 도달할 때까지 측정된 run-to-failure 데이터 Validation: EOL에 도달하지 않았고 결함이 진행 중인 상태까지만 제공 데이터: 진동 25.6kHz, 4채널, 10분마다 1분 측정 운전 조건: 약 700-950 rpm, 1시간 단위 변경, 다량 노이즈 포함 항목 값 베어링 NSK 30306 Tapered Roller 축방향/반경방향 하중 15 kN / 10 kN EOL 조건 1 하우징 온도 >= 200°C EOL 조건 2 회전 토크 <= -20 Nm 제출 Validation 각 베어링 RUL(초) 중요 현재 코드의 Test1~Test6는 공식 문맥상 Validation Set이다. 즉, 각 베어링의 마지막 관측 시점 이후 남은 수명을 맞히는 문제다.

## ppt/slides/slide30.xml

Research TCN-Transformer와 동역학 피처 리서치 반영 30 KSPHM-KIMM 2026 Bearing RUL Prediction 최근 베어링 RUL 연구는 TCN-Transformer 계열 구조를 적극적으로 활용한다. 공통 아이디어는 명확하다. TCN은 짧은 구간의 국소 열화 패턴을 잡고, Transformer는 긴 시간 의존성을 본다. 성능이 좋은 연구들은 단일 진동값보다 시간·주파수·시간-주파수 피처를 함께 사용한다. 최근 흐름은 현재값 예측을 넘어, 열화 속도·가속도·궤적 유사도 같은 동역학 정보를 넣는 방향이다. 우리의 현재 위치 이미 TCN + Transformer 구조를 사용 부족한 부분 명시적 동역학 피처가 약함 다음 개선 v25 = v24 + dynamics + DTW

## ppt/slides/slide31.xml

Research 왜 MADA/Domain Adaptation은 아직 없는가? 31 KSPHM-KIMM 2026 Bearing RUL Prediction MADA란? Multi-Adversarial Domain Adaptation. Source(Train)와 Target(Validation)의 feature 분포 차이를 줄이는 domain adaptation 방법이다. 왜 중요하나? 베어링마다 결함 모드, 노이즈, 채널 감도, 열화 속도가 다르다. Train에서 배운 feature가 Validation에 그대로 맞지 않을 수 있다. 왜 v24에는 없나? v24는 우선 LOBO 검증 점수와 제출값 안정화를 목표로 했다. MADA는 target pseudo label/stage 정의가 필요해 추가 검증 없이 넣으면 불안정할 수 있다. 적용 방법 RUL 회귀 문제이므로 class label 대신 HI 기반 degradation stage를 pseudo class로 나눈 뒤 stage-wise domain alignment를 적용한다. 후보 기법 DANN, MMD/CORAL, MADA, multi-source domain adaptation. Train1~4를 source domains로 보고 Validation을 target으로 둔다. v26 방향 v25에서 동역학 피처를 먼저 안정화한 뒤, v26에서 HI-stage 기반 MADA를 실험하는 순서가 안전하다.

## ppt/slides/slide32.xml

Next v25 제안: Test5를 검증하는 동역학 모델 32 KSPHM-KIMM 2026 Bearing RUL Prediction 추가 피처 의미 d_HI_1 / d_HI_5 HI 단기/중기 변화량 slope_HI_5 / slope_HI_10 최근 열화 속도 acc_HI 열화 가속도 roll_std_HI 불안정성/충격적 변화 d_env_kurt 충격성 결함 진행 DTW distance Train EOL 궤적과 유사도 목표 Test5의 높은 HI가 실제 EOL 근접 신호인지, 일시적 노이즈인지 구분한다. 전략 v24를 폐기하지 않고, 동역학 피처와 DTW로 제출값을 검증·보정한다. submission 후보 = v24 combined / v19 combined / v25 dynamics

## ppt/slides/slide33.xml

Next 이후 시도할 방향 33 KSPHM-KIMM 2026 Bearing RUL Prediction 1. Test5 동역학 검증 HI가 높다는 사실만으로는 부족하다. 최근 HI 기울기, 가속도, envelope kurtosis 변화, 채널 에너지 변화가 함께 악화되는지 확인한다. 2. DTW 궤적 매칭 Test 베어링의 HI 궤적이 Train1~4의 어느 열화 구간과 가장 유사한지 비교한다. 특히 EOL 직전 구간과 닮았는지 확인한다. 3. v25 모델 v22 피처에 동역학 피처를 추가하고, 기존 TCN-Transformer/BiLSTM 구조는 유지한다. 모델을 키우기보다 입력 정보를 개선한다. 4. 제출 후보 비교 v19, v24, v25를 같은 LOBO 기준으로 비교하고, Test5 예측이 얼마나 민감한지 시나리오 분석한다. 5. 보고서 강화 단순 딥러닝 구조가 아니라, 왜 특정 베어링을 EOL 근접으로 판단했는지 물리·신호처리 근거를 제시한다. 6. Domain Adaptation MADA/DANN/MMD는 v26 후보로 둔다. HI-stage pseudo label이 안정화된 뒤 적용한다.

## ppt/slides/slide34.xml

Research Google Scholar 기반 관련 연구 34 KSPHM-KIMM 2026 Bearing RUL Prediction 연구 핵심 아이디어 우리 적용 포인트 Cao et al., 2024 TCN-Transformer TCN + Transformer로 bearing RUL 예측 현재 TFTModel 구조와 유사 Peng et al., 2023 Local-enhancing Transformer TCN attention으로 Transformer의 국소 패턴 보완 local degradation feature 강화 Tang et al., 2024 Parallel TCN + Transformer TCN branch와 Transformer branch 병렬 결합 v25/v26 아키텍처 후보 Jin et al., 2025 Sensors TCN-Transformer 진동 신호 기반 TCN-Transformer RUL v24 설명 근거 Cao et al., 2024 Multi-domain + TCN 시간·주파수·시간-주파수 피처 결합 동역학 피처 추가 근거 DTW / phase-space warping studies 열화 궤적 유사도 기반 RUL 추정 Test5 sanity check MADA / DANN / MMD studies Source-target feature distribution alignment v26 domain adaptation 후보

## ppt/slides/slide35.xml

Research Google Scholar: Domain Adaptation / MADA 계열 35 KSPHM-KIMM 2026 Bearing RUL Prediction 연구 기법 우리 적용 포인트 Tian et al., 2023 MDA-LETCN Multistage degradation + domain adaptation + TCN 열화 stage별 분포 보정 아이디어 Wen et al., 2021 BGRU-DANN Data-driven RUL prediction based on domain adaptation Source/target feature alignment 기본형 Liu & Gryllias, 2020 UDA-DANN Unsupervised domain adaptation for rolling bearing RUL Validation label 없이 적응하는 구조 Ding et al., 2022 Multi-source DA Operating condition별 multi-source adaptation, MMD/CORAL Train1~4를 multi-source로 보는 근거 Cao et al., 2026 Stage-aware DA Transformer Degradation stage-aware adversarial domain adaptation HI-stage pseudo label + Transformer 후보 Lu et al., 2025 Dynamic hybrid DA Cross-domain rolling bearing RUL + attention contrastive learning v26 고급 후보 Hu et al., 2022 Wasserstein DA Working condition 차이를 Wasserstein distance로 보정 운전조건 변화 대응 Miao et al., 2022 Sparse DA Different working conditions under sparse domain adaptation 작은 데이터 조건 대응 Zhang et al., 2025 JDA Transformer Joint domain-adaptive Transformer for bearing RUL TCN/Transformer 이후 DA 결합 후보 정리: v24에는 아직 MADA가 없지만, Train1~4와 Validation 간 분포 차이를 줄이기 위한 v26 후보로 명확히 가치가 있다.

## ppt/slides/slide36.xml

Summary 발표용 핵심 문장 36 KSPHM-KIMM 2026 Bearing RUL Prediction 문제 정의 본 과제는 Train 베어링의 run-to-failure 데이터를 기반으로, EOL 전 상태까지만 공개된 Validation 베어링의 남은 수명을 초 단위로 예측하는 문제입니다. 피처 설계 진동 신호에서 충격이 잘 보이는 대역을 찾고, envelope order 피처와 4채널 대칭 피처를 추출한 뒤, DTC-VAE로 건강지수 HI를 구성했습니다. 모델 구조 최근 10개 측정의 피처 시계열을 TCN-Transformer와 BiLSTM 다중 seed 앙상블에 입력하여 RUL을 예측했습니다. 최종 판단 v24는 HI가 높은 Test5만 EOL 근접으로 판단하는 보수적 ensemble이며, 늦은 예측 패널티가 큰 공식 평가식에 맞춘 전략입니다.

## ppt/slides/slide37.xml

References 참고 방법론 및 출처 37 KSPHM-KIMM 2026 Bearing RUL Prediction KSPHM-KIMM 2026 공식 Notion 및 KIMM Data Platform: 데이터 조건, Train/Validation 정의, EOL 조건 W. Cao et al., 2024, A remaining useful life prediction method for rolling bearing based on TCN-Transformer H. Peng et al., 2023, Local enhancing Transformer with temporal convolutional attention mechanism for bearings RUL prediction Y. Tang et al., 2024, RUL prediction of rolling bearings based on TCN and Transformer in parallel X. Jin et al., 2025, Remaining Useful Life Prediction for Rolling Bearings Based on TCN-Transformer Networks Using Vibration Signals X. Cao et al., 2024, RUL prediction of rolling bearing based on multi-domain mixed features and temporal convolutional networks DTW, phase-space warping, physics-informed Transformer 기반 bearing degradation tracking/RUL 연구 M. Tian et al., 2023, multistage degradation RUL prediction of wind turbine generator bearings based on domain adaptation B. Wen et al., 2021, Data-driven remaining useful life prediction based on domain adaptation C. Liu and K. Gryllias, 2020, Unsupervised domain adaptation based RUL prediction of rolling element bearings Y. Ding et al., 2022, Transfer learning for RUL prediction across operating conditions based on multisource domain adaptation L. Cao et al., 2026, Staged domain adaptation with Transformer for bearing RUL prediction

## ppt/slides/slide38.xml

결론 v24는 공식 평가식에 맞춘 제출 후보입니다 핵심 리스크는 Test5의 EOL 근접성 판단이며, 이후에는 동역학 피처와 DTW 궤적 검증으로 이 판단을 보강하겠습니다. v24 = robust baseline(v17) + EOL-sensitive model(v22) + HI switch 현재 제출 후보 submission_v24_v17v22_combined.xlsx 백업 후보 submission_v19_blend_combined.xlsx 다음 실험 v25 dynamics + DTW

## ppt/slides/slide4.xml

Metric 평가 지표: 비대칭 RUL 점수 04 KSPHM-KIMM 2026 Bearing RUL Prediction Er = 100 × (ActRUL - PredRUL) / ActRUL 왜 비대칭인가? PredRUL > ActRUL이면 실제보다 오래 남았다고 본 늦은 예측이다. 산업 현장에서는 고장 직전 장비를 계속 돌리게 되므로 더 위험하다. 상황 오차 점수 완벽 0% 1.000 늦은 예측 20% 0.500 이른 예측 50% 0.500 늦은 예측 10% 0.707 이른 예측 25% 0.707 해석 늦은 예측이 이른 예측보다 약 2.5배 가혹하다. 그래서 EOL 의심이 강하면 보수적으로 짧게 예측하는 전략이 합리적이다.

## ppt/slides/slide5.xml

Data 600s의 의미 05 KSPHM-KIMM 2026 Bearing RUL Prediction 측정 간격 데이터는 10분마다 1분씩 측정된다. 따라서 측정 인덱스 간 시간 간격은 600초다. Train 라벨 코드에서는 마지막 측정 이후 다음 600초 지점을 EOL로 가정한다. 그래서 마지막 RUL은 대체로 600s가 된다. 예측 하한 v22/v24는 예측값을 max(pred, 600)으로 자른다. 너무 작은 초 단위 예측을 막는 안전장치다. eol = last_measurement_time + 600 pred = max(pred, 600) 따라서 Test5 = 644s는 '공식 최소값'이 아니라, 모델이 '다음 10분 안팎 EOL'로 판단했다는 의미다.

## ppt/slides/slide6.xml

Basics Kurtosis: 충격성 결함을 보는 기본 통계 06 KSPHM-KIMM 2026 Bearing RUL Prediction Kurtosis란? 신호가 얼마나 '뾰족하게 튀는지'를 보는 통계량이다. 평균 근처에 잔잔한 값만 있으면 낮고, 가끔 큰 충격이 튀면 높다. kurtosis = E[(x - μ)^4] / σ^4 베어링에서의 의미 균열·박리·찍힘이 생기면 회전할 때마다 충격이 발생한다. 이때 kurtosis가 상승할 수 있다. 상태 파형 느낌 Kurtosis 정상 잔잔하고 균일 낮음 초기 결함 가끔 퍽 튐 상승 심한 결함 충격 반복 높음 또는 불안정

## ppt/slides/slide7.xml

Basics Fast Kurtogram: 충격이 잘 보이는 주파수 대역 찾기 07 KSPHM-KIMM 2026 Bearing RUL Prediction 정확한 표현 우리가 코드에서 fast_kurt라고 부르는 것은 단순 '빠른 kurtosis'가 아니라 Fast Kurtogram 스타일의 대역 선택이다. 무엇을 하나? 여러 주파수 대역을 후보로 두고, 각 대역에서 충격성(kurtosis)이 가장 큰 곳을 고른다. 왜 필요한가? 베어링 결함 충격은 원신호 전체보다 특정 공진 대역에서 더 잘 보인다. 좋은 대역을 먼저 고르면 SNR이 좋아진다. candidate bands: 0.5-2k, 1-4k, 2-6k, 3-8k, 4-10k, 5-12k Hz

## ppt/slides/slide8.xml

Basics Envelope와 Hilbert Transform 08 KSPHM-KIMM 2026 Bearing RUL Prediction Envelope 진동 신호의 '겉선' 또는 '포락선'이다. 빠르게 흔들리는 원신호 안에서 충격 반복 패턴을 더 잘 보이게 한다. Hilbert Transform 실수 신호 x(t)에 90도 위상 차이 성분을 만들어 analytic signal을 구성하는 수학 도구다. 왜 쓰나? 베어링 결함은 반복 충격이다. Envelope FFT를 보면 BPFI/BPFO/BSF/FTF 주변 에너지가 더 잘 드러난다.

## ppt/slides/slide9.xml

Basics Order와 베어링 결함 주파수 09 KSPHM-KIMM 2026 Bearing RUL Prediction RPM이 바뀌면 결함 주파수의 Hz 위치도 같이 이동한다. Order는 주파수를 회전주파수로 나눈 좌표라 RPM 변화에 덜 흔들린다. 코드에서는 각 결함 order와 2배·3배 고조파 주변 ±0.15 대역 에너지를 합산한다. Order = frequency / (RPM / 60) 결함 Hz @1000RPM Order 의미 BPFI 140 8.40 내륜 BPFO 93 5.58 외륜 BSF 78 4.68 롤러/볼 FTF 6.7 0.40 케이지