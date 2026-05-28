# v20 Pipeline - 4-Channel Feature Expansion Trial

## 목적

v20은 단일 채널 중심 v17/v18의 한계를 보완하기 위해 4채널 피처를 확장한 실험 버전이다. 특히 특정 채널에서만 강하게 나타나는 Train4류 고장 패턴 대응을 목표로 했다.

## 핵심 아이디어

- 4채널 각각에서 동일한 time/frequency/envelope/order 피처 추출
- 채널 간 correlation, energy ratio 등 다채널 관계 피처 추가
- 기존 TFT/BiLSTM 시계열 모델 구조 유지

## 전처리

1. 4개 채널 각각에 대해 Fast Kurtogram 스타일 대역 선택
2. 각 채널별 band-pass 및 Hilbert envelope 계산
3. 각 채널별 order spectrum 계산
4. 채널 간 correlation 및 energy distribution 계산

## 피처

- 채널별 `rms`, `std`, `kurt`, `skew`, `peak`, `crest`, `p2p`
- 채널별 `env_rms`, `env_kurt`
- 채널별 BPFI/BPFO/BSF/FTF energy/SNR/harmonic ratio
- 다채널 통합: `rms_multi`, `std_multi`, `peak_multi`
- 채널 관계: `corr_ij`, `energy_max`, `energy_min`, `energy_ratio`, `energy_std`

## 모델 및 학습

- DTC-VAE HI
- TFT 5 seeds + BiLSTM 5 seeds
- Gaussian noise + Mixup
- LOBO 검증

## 결과와 역할

v20은 채널 정보 확장 자체는 유효했지만, EOL 근처 Last 성능이 v18/v22만큼 안정적이지 않았다. 이후 v22는 v20에서 입증된 채널 대칭 피처 아이디어를 가져오되, v18의 EOL 강화 학습 설정을 결합했다.
