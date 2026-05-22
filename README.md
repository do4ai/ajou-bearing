# Ajou Bearing RUL Prediction

KSPHM-KIMM 2026 기계 데이터 챌린지 — NSK 30306 테이퍼 롤러 베어링 잔여수명(RUL) 예측

## 챌린지 개요

- **목표**: 가변 운전조건 하에서 베어링의 잔여수명(Second) 예측
- **데이터**: 진동(25.6kHz, 4채널) + 운전조건(RPM, 토크, 온도), TDMS 포맷
- **평가**: 비대칭 점수 (늦은 예측 2.5x 패널티)
- **일정**: 예비 제출 6/1~6/5, 최종 제출 6/8

## 프로젝트 구조

```
├── shared/utils.py              # 공통 유틸 (asym_score, load_bearing)
├── data/
│   ├── raw/{Train1..Val2}/      # 베어링별 vibration.npy + operating.csv
│   ├── synthetic_generator.py   # 합성 데이터 생성기
│   └── convert_tdms.py          # TDMS → numpy 변환 스크립트
├── 01_경민팀_OrderTracking/     # Order Tracking + CNN-BiLSTM
│   ├── pipeline.py
│   ├── models/
│   └── results/
├── 02_태환팀_SpectrogramCNN/    # VMD + STFT Spectrogram + 2D CNN-LSTM
│   ├── pipeline.py
│   ├── models/
│   └── results/
├── 03_교수설계_Ensemble/        # Fast Kurtogram + DTC-VAE + XGB+LSTM+GPR 앙상블
│   ├── pipeline.py
│   ├── models/
│   └── results/
├── docs/challenge_info.md       # 챌린지 공식 정보
├── compare_results.py           # 결과 비교 스크립트
└── research/                    # 연구 자료
```

## 3가지 방법론

### 1. 경민팀 — Order Tracking + CNN-BiLSTM
- Bandpass 필터링 → 각도 리샘플링 → Order Spectrum 추출
- 16개 피처(RMS, Kurtosis, Order 에너지 등) + Health Indicator
- CNN-BiLSTM + Attention (SEQ=10)

### 2. 태환팀 — VMD + STFT Spectrogram + 2D CNN-LSTM
- VMD 근사 (Bandpass 분해) → 최고 Kurtosis 모드 선택
- STFT Spectrogram (32x32) → Baseline 차감
- 2D CNN 인코더 + LSTM + Attention (SEQ=8)

### 3. 교수설계 — Fast Kurtogram + DTC-VAE + 앙상블
- Fast Kurtogram → 최적 대역 필터링 → Order 에너지 피처
- DTC-VAE로 단조 증가 Health Indicator 학습
- CUSUM 기반 FPT 탐지 → 가중 RUL 타겟
- XGBoost + LSTM + GPR 앙상블 (점수 기반 가중치)

## 현재 성능 (합성 데이터, LOBO)

| 방법론 | 평균 RMSE (s) | 평균 AsymScore |
|--------|--------------|----------------|
| 경민팀 OrderTracking | ~8,800 | **0.72** |
| 태환팀 SpectrogramCNN | ~10,300 | **0.63** |
| 교수설계 Ensemble | ~7,500 | **0.55** |

## 사용법

```bash
# 데이터 준비 (TDMS → numpy)
python data/convert_tdms.py --input /path/to/Train --explore  # 구조 확인
python data/convert_tdms.py --input /path/to/Train            # 변환

# 파이프라인 실행
python 01_경민팀_OrderTracking/pipeline.py
python 02_태환팀_SpectrogramCNN/pipeline.py
python 03_교수설계_Ensemble/pipeline.py

# 결과 비교
python compare_results.py
```

## 평가 공식

```
Er = 100 × (ActRUL - PredRUL) / ActRUL     (% 오차)

A  = exp(-ln(0.5) × Er/20)  if Er ≤ 0     (늦은 예측, 가혹)
A  = exp(+ln(0.5) × Er/50)  if Er > 0      (이른 예측, 관대)

최종 점수 = mean(A)  ∈ (0, 1], 완벽 = 1.0
```

## 베어링 규격 — NSK 30306

| 파라미터 | Order | Hz @ 1000 RPM |
|----------|-------|---------------|
| BPFI     | 8.40  | 140           |
| BPFO     | 5.58  | 93            |
| BSF      | 4.68  | 78            |
| FTF      | 0.40  | 6.7           |

## 의존성

```
numpy, pandas, scipy, scikit-learn, torch, xgboost, nptdms, matplotlib
```
