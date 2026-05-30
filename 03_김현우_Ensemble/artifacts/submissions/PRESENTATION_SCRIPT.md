# 발표 스크립트 — KSPHM-KIMM 2026 베어링 RUL 예측 (5분)

> 평가 = 발표 30% + 창의성 10% + 합리성 10% (방법론·발표 50%).
> 핵심 전략: "train-based 원칙 + 비대칭 페널티 정합 + 위험 분산"을 명확한 스토리로.

---

## [0:00–0:30] 도입 — 문제와 우리의 차별점

"NSK 30306 테이퍼 롤러 베어링 6대의 잔여수명(RUL)을 초 단위로 예측하는 과제입니다.
저희 팀의 핵심 원칙은 한 가지입니다 — **모든 예측값은 Train data로 학습된 모델의 회귀 결과여야 한다.**
임의로 '안전치 6000초' 같은 숫자를 박는 것은 챌린지의 본질, 즉 데이터로 RUL을 예측하라는 요구에 어긋납니다.
저희는 이 원칙을 끝까지 지켰습니다."

## [0:30–1:15] 평가식의 비대칭성 — 학습에 직접 반영

"평가식을 먼저 보겠습니다. 오차 Er은 퍼센트이고, 늦은 예측(실제보다 길게)은 20으로 나누고,
이른 예측(짧게)은 50으로 나눕니다. 즉 **늦은 예측이 2.5배 더 가혹**합니다.
산업 현장에서 고장 직전 장비를 계속 돌리는 게 더 위험하기 때문입니다.

저희는 이 비대칭성을 후처리가 아니라 **학습 손실과 예측 산출 양쪽에 직접 반영**했습니다.
- 학습: asymmetric MSE penalty 5배 + EOL 구간 50배 가중
- 예측: 단일 점 예측을 argmax_p E[A(p,R)] 로 비대칭 기대값 최대화"

## [1:15–2:15] 파이프라인 — 신호에서 RUL까지

"파이프라인은 7단계입니다.
1. 4채널 진동 신호를 Fast Kurtogram으로 충격이 잘 보이는 대역을 찾고,
2. Hilbert envelope으로 BPFI/BPFO/BSF/FTF 결함 주파수 에너지를 추출합니다.
3. 채널 대칭 피처 — 4채널의 max/min/range/std — 로 한 채널만 고장나도 잡습니다.
4. DTC-VAE로 건강지수 HI를 만들고,
5. 동역학 피처 — 변화 속도, 가속도 — 를 더합니다.
6. 그리고 핵심: Train의 HI 구간별 실제 RUL 분포를 prior로 사용합니다.
7. 최종은 TCN-Transformer와 BiLSTM 앙상블, 그리고 베어링별 최적 선택입니다."

## [2:15–3:15] 핵심 통찰 — Train HI vs RUL 분포 + Test5 anomaly

"저희가 발견한 가장 중요한 통찰입니다.
**모든 Test 베어링은 정확히 8.17시간, 50개 측정에서 관측이 끝납니다.** 동일한 관측 윈도우죠.
그런데 같은 8.17시간 시점에서 Train 베어링들의 HI는 모두 0.42~0.51로 비슷합니다.
**그런데 Test5만 HI가 0.944입니다.** 어떤 Train 베어링보다도 훨씬 빠르게 열화한 겁니다.

이건 Test5가 train 분포 밖의 anomaly, 즉 비정상적으로 빠른 고장 진행임을 뜻합니다.
→ Test5는 짧은 RUL을 예측하는 것이 데이터로 정당화됩니다. 저희는 644초로 예측했습니다.
반대로 Test3은 HI 0.165로 정상보다 느려서, 긴 RUL을 예측합니다."

## [3:15–4:15] 방법론 다양성 + 위험 분산

"저희는 단일 모델에 의존하지 않고 여러 train-based 방법을 만들었습니다.
- **17_AsymOptimal**: HI 상태 KNN + 비대칭 페널티 직접 최적화
- **19_EOLProgression**: HI 곡선을 fitting해 진행도를 역산, LOBO 0.75 달성
- **28_EOL Specialist**: 고장 직전 구간만 전문으로 학습한 회귀기
- **5_HIBlend**: 검증된 TCN-Transformer 앙상블, LOBO 0.712

이들을 베어링별로 sensitivity 분석해 가장 robust한 예측을 선택합니다.
그리고 흥미로운 발견 — **LOBO 점수와 sensitivity 점수가 anti-correlated**입니다.
LOBO는 train의 마지막 600초 라벨에 fit되고, sensitivity는 HI 구간 prior 가정에 의존하기 때문입니다.
저희는 이 두 관점이 모두 합리적인 후보를 1순위로, 나머지를 백업으로 두어 **위험을 분산**했습니다."

## [4:15–5:00] 결론 — 제출 전략

"최종 제출은 3개의 다른 가설로 구성됩니다.
- **1순위**: 베어링별 최적 혼합 (sensitivity 0.488)
- **백업1**: 검증된 5_HIBlend (LOBO 0.712)
- **백업2**: EOL progression physics (LOBO 0.75)

세 후보 모두 임의 숫자 없이 train data로만 만들어졌고, 비대칭 평가식에 정합합니다.
저희의 기여는 단순히 높은 점수가 아니라, **데이터에 충실하면서도 평가식의 비대칭성을 끝까지 존중한 설명 가능한 파이프라인**입니다.
감사합니다."

---

## 예상 Q&A

**Q1. Test5를 644초로 예측한 근거는?**
> HI=0.944로 가장 높고, 동일 8.17시간 관측에서 어떤 train 베어링보다 빠르게 열화했습니다 (train은 모두 ~0.5).
> DTW로도 Train1의 EOL 직전(1200초)과 가장 유사합니다. 비대칭 페널티상 짧은 예측이 robust합니다.

**Q2. 왜 단일 best 모델이 아니라 베어링별 선택인가?**
> 베어링마다 열화 단계가 달라 (HI 0.16~0.94) 최적 모델이 다릅니다.
> mid-life(Test1,2,4,6)는 EOL specialist, 초기(Test3)는 trajectory KNN, 고장임박(Test5)은 HI-blend가 robust합니다.

**Q3. LOBO 점수가 낮은 후보(28_eol 0.054)를 왜 쓰나?**
> LOBO는 train 마지막 측정(600초)만 평가해 EOL specialist에 불리합니다.
> 실제 Test는 EOL 미도달이라 mid-life 예측이 중요하고, sensitivity 분석에서 28_eol이 mid-life에 robust합니다.

**Q4. 과적합 방지는?**
> 모든 calibration은 LOBO out-of-fold로만 수행. Test 데이터에 맞춘 hand-tuning은 final 후보에서 제외.
> Fixed feature, fixed threshold, fixed KNN/window만 적용 (docs/FINAL_ANTI_OVERFIT_PROTOCOL.md).

**Q5. 다른 팀(CNN-BiLSTM, Spectrogram) 대비 차별점은?**
> 저희는 단일 end-to-end 모델 대신 신호처리 + HI + train-based prior + 비대칭 정합 ensemble을 결합.
> 특히 평가식 비대칭성을 학습·예측·calibration 전 단계에 일관되게 반영한 점이 차별점입니다.
