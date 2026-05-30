# 발표 리허설 가이드 — KSPHM-KIMM 2026 (5분 + Q&A)

> PRESENTATION_SCRIPT.md를 슬라이드에 매핑 + 시간 큐. 발표 30% 점수용.
> 자료: v24 본편 38장 + v26 addendum 14장. 5분 발표는 핵심 12장만 사용.

## 5분 발표 슬라이드 시퀀스 (핵심 12장)

| 시간 | 슬라이드 | 발표 포인트 | 큐 문장 |
|------|---------|-------------|---------|
| 0:00–0:30 | 본편 s1 (Title) | 문제 + train-based 원칙 | "모든 예측은 train data로 학습된 회귀값" |
| 0:30–1:15 | 본편 s4 (Metric) + addendum A02 | 비대칭 페널티 2.5× → 학습/예측에 직접 반영 | "후처리가 아니라 학습 손실에" |
| 1:15–2:00 | addendum A02 파이프라인 + 본편 s10 (Features) | 7단계: 신호→HI→dynamics→prior | "Fast Kurtogram→envelope→채널대칭→DTC-VAE HI" |
| 2:00–2:45 | **addendum A13 (HI-Velocity Anomaly)** | Test5 핵심 근거 | "동일 8.17h에 train 모두 HI~0.5, Test5만 0.944" |
| 2:45–3:15 | addendum A08 (Train HI-RUL 분포) | HI-band prior | "HI 구간별 실제 RUL 분포를 prior로" |
| 3:15–4:00 | addendum A04 (PerBearing) + A07 (LOBO vs Sens) | 베어링별 선택 + 검증 | "selection LOBO 0.519, naive 능가 = overfit 아님" |
| 4:00–4:30 | addendum A11/A12 (heatmap, comparison) | 시각화 | "베어링별 best, 위험 분산" |
| 4:30–5:00 | addendum A10 (제출 전략) | 1순위 + 백업 다양화 | "sensitivity / LOBO / physics 세 가설" |

## 시간 배분 원칙
- 도입·원칙 (0:00–1:15): 45초 — 빠르게, 강한 한 문장
- 방법 (1:15–2:00): 45초 — 파이프라인 흐름만, 디테일 X
- **Test5 anomaly (2:00–2:45): 45초 — 가장 강조. 발표의 클라이맥스**
- 검증 (2:45–4:00): 75초 — LOBO 0.519 generalize 입증이 합리성 점수 핵심
- 제출 전략 (4:00–5:00): 60초 — 위험 분산으로 마무리

## 발표자 노트 (강조/주의)

### 반드시 말할 3가지 (차별점)
1. **"임의 숫자를 박지 않았다"** — 챌린지 정신 준수 (창의성·합리성)
2. **"Test5는 train 분포 밖 anomaly"** — 데이터 기반 의사결정 (우수성)
3. **"selection 방법을 LOBO로 검증했다 (0.519 > naive)"** — overfit 아님 (합리성)

### 피해야 할 것
- 모델 아키텍처 디테일 장황하게 (TCN dilation 등) → 흐름 끊김
- "다른 팀보다 높다" 식 비교 (협업팀이므로 무의미)
- LOBO 점수만 자랑 (Test 실제 RUL ≠ train 600s 라벨임을 인정해야 신뢰)

### Q&A 대비 (PRESENTATION_SCRIPT.md 5개 + 추가)
- Q6. "Test3를 13.6시간으로 길게 본 근거?" → HI=0.16은 train low-band(RUL 42000~82200s)와 매칭, 48900s는 그 분포의 비대칭 최적값.
- Q7. "예비/최종 제출 차이?" → 예비에서 점수 확인 후 최종 조정. 백업 3종으로 가설 검증.
- Q8. "임의 clamp 없이 어떻게 폭주 방지?" → 600s 물리 하한(측정 간격)만. 나머지는 train RUL 분포가 자연 상한 제공.

## 리허설 체크
- [ ] 5분 내 완료 (타이머)
- [ ] Test5 anomaly 슬라이드에서 청중 시선 집중 확인
- [ ] LOBO 0.519 검증 결과 숫자 정확히 암기
- [ ] Q&A 8개 30초 내 답변 연습
- [ ] 백업 슬라이드(본편 s19~37 상세) 위치 숙지 (심화 질문 대비)
