# Bearing RUL Literature Review for KSPHM-KIMM 2026

이 문서는 발표자료와 후속 실험 설계를 위해 Google Scholar / 공식 웹에서 확인한 관련 연구를 정리한 것이다. PDF가 공개 접근 가능한 경우 `research/`에 저장했고, 접근 제한 또는 anti-bot으로 PDF 저장이 실패한 경우 원문 링크와 활용 포인트를 남긴다.

## 현재 로컬 PDF

| 파일 | 상태 | 비고 |
|---|---:|---|
| `papers/public/TCN-TFT-BiLSTM_RUL_2024.pdf` | 정상 | 기존 보유 자료 |
| `papers/public/PINN-Attention-RUL_arXiv2024.pdf` | 정상 | 기존 보유 자료 |
| `papers/public/DeepLearning-MotorBearing-DoubleLoss_2024.pdf` | 정상 | 기존 보유 자료 |
| `papers/public/LLM-Transfer-RUL_arXiv2025.pdf` | 정상 | 기존 보유 자료 |
| `papers/public/Adversarial_DomainAdaptation_RUL_SciReports_2022.pdf` | 정상 | adversarial DA 기반 RUL |
| `papers/public/Jin_2025_Sensors_TCN_Transformer_Vibration_RUL.pdf` | 정상 | PPT References: Sensors TCN-Transformer |
| `papers/public/Cao_2024_AppliedSciences_Multidomain_TCN_RUL.pdf` | 정상 | PPT References: multi-domain features + TCN |
| `papers/public/Tian_2023_AppliedSciences_MDA_LETCN_DomainAdaptation_RUL.pdf` | 정상 | PPT References: MDA-LETCN |
| `papers/public/Liu_Gryllias_2020_UDA_DANN_Bearing_RUL.pdf` | 정상 | PPT References: UDA-DANN |
| `papers/public/Liu_2025_PLOS_VMD_BiLSTM_CBAM_RUL.pdf` | 정상 | 추가 VMD/BiLSTM-CBAM 참고 |

다음 공개 논문들은 사이트가 PDF 대신 HTML/redirect를 내려줘서 로컬 PDF로 저장하지 않았다. 원문 URL과 적용 포인트는 아래 표에 유지한다.

- PeerJ 2021 data-driven RUL prediction based on domain adaptation

## TCN / Transformer / Multi-domain Feature

| 논문 | 연도 | 링크 | 핵심 | 우리 적용 |
|---|---:|---|---|---|
| A remaining useful life prediction method for rolling bearing based on TCN-Transformer | 2024 | https://ieeexplore.ieee.org/abstract/document/10758686/ | TCN + Transformer 결합으로 rolling bearing RUL 예측 | 현재 `TFTModel`이 TCN + Transformer Encoder 구조라 방법론 근거로 사용 |
| Bearing remaining useful life prediction based on TCN-Transformer model | 2023 | https://ieeexplore.ieee.org/abstract/document/10295609/ | GRU/Transformer 단독 대비 TCN-Transformer 비교 | v25/v26에서 TCN-Transformer branch 설명 근거 |
| Local enhancing Transformer with temporal convolutional attention mechanism for bearings RUL prediction | 2023 | https://ieeexplore.ieee.org/abstract/document/10177813/ | Transformer의 local pattern 약점을 TCN attention으로 보완 | 동역학/국소 열화 패턴 강조 근거 |
| Remaining useful life prediction of rolling bearings based on time convolutional network and transformer in parallel | 2024 | https://iopscience.iop.org/article/10.1088/1361-6501/ad73ee/meta | TCN과 Transformer 병렬 결합 | 현재 구조 이후 병렬 branch 실험 후보 |
| Remaining Useful Life Prediction for Rolling Bearings Based on TCN-Transformer Networks Using Vibration Signals | 2025 | https://www.mdpi.com/1424-8220/25/11/3571 | 진동 신호 기반 TCN-Transformer | 발표자료 레퍼런스, v24 설명 근거 |
| Remaining useful life prediction of rolling bearing based on multi-domain mixed features and temporal convolutional networks | 2024 | https://www.mdpi.com/2076-3417/14/6/2354 | 시간/주파수/시간-주파수 multi-domain feature + TCN | v25 동역학/다중영역 피처 추가 근거 |
| Prediction of contact fatigue performance degradation trends based on multi-domain features and TCN | 2023 | https://www.mdpi.com/1099-4300/25/9/1316 | multi-domain features로 열화 trend 예측 | feature engineering 근거 |

## DTW / Trajectory Matching / Phase-space Warping

| 논문 | 연도 | 링크 | 핵심 | 우리 적용 |
|---|---:|---|---|---|
| Assessment of rolling element bearing degradation based on Dynamic Time Warping, KRR and SVR | 2023 | https://www.sciencedirect.com/science/article/pii/S0003682X23001871 | DTW feature로 bearing degradation trend 추적 | Test5가 Train EOL 궤적과 닮았는지 sanity check |
| Multivariate phase space warping-based degradation tracking and RUL prediction of rolling bearings | 2024 | https://ieeexplore.ieee.org/abstract/document/10436418 | 다변량 phase-space warping으로 열화 추적 | v25 trajectory distance 후보 |
| Remaining useful life prediction based on segmented relative phase space warping and particle filter | 2022 | https://ieeexplore.ieee.org/abstract/document/9919217 | phase-space warping + particle filter | 작은 데이터에서 similarity 기반 보조 예측 후보 |
| Prediction of remaining useful life by data augmentation technique based on DTW | 2020 | https://www.sciencedirect.com/science/article/pii/S0888327019307071 | DTW 기반 data augmentation | 데이터 부족 보완 아이디어 |

## Domain Adaptation / MADA / DANN / MMD / CORAL

| 논문 | 연도 | 링크 | 핵심 | 우리 적용 |
|---|---:|---|---|---|
| A novel method for multistage degradation predicting RUL of wind turbine generator bearings based on domain adaptation | 2023 | https://www.mdpi.com/2076-3417/13/22/12332 | multistage degradation + domain adaptation + TCN(MDA-LETCN) | HI-stage pseudo label 기반 MADA 후보 |
| Data-driven remaining useful life prediction based on domain adaptation | 2021 | https://peerj.com/articles/cs-690/ | BGRU-DANN 등 domain-adaptive RUL 구조 | Train/Validation feature distribution alignment 근거 |
| Unsupervised domain adaptation based RUL prediction of rolling element bearings | 2020 | https://pdfs.semanticscholar.org/88d3/0284ffcea288943c7b2500601ffc5310f1c4.pdf | DANN 기반 unsupervised DA for bearing RUL | Validation label 없이 적용 가능한 DA 후보 |
| Transfer learning for RUL prediction across operating conditions based on multisource domain adaptation | 2022 | https://ieeexplore.ieee.org/abstract/document/9723508/ | multi-source domain adaptation, MMD/CORAL | Train1~4를 multi-source domain으로 보는 근거 |
| Cross-condition and cross-platform RUL estimation via adversarial-based domain adaptation | 2022 | https://www.nature.com/articles/s41598-021-03835-2 | adversarial DA 기반 cross-condition RUL | 로컬 PDF 저장됨 |
| Remaining useful life estimation of bearings under different working conditions via Wasserstein distance-based weighted DA | 2022 | https://www.sciencedirect.com/science/article/pii/S0951832022001806 | working condition 차이를 Wasserstein distance로 보정 | 운전조건 변화 대응 후보 |
| A sparse domain adaptation network for RUL prediction of rolling bearings under different working conditions | 2022 | https://www.sciencedirect.com/science/article/pii/S0951832021007353 | sparse DA for bearing RUL | 작은 데이터 조건 대응 후보 |
| Staged domain adaptation with Transformer for bearing RUL prediction | 2026 | https://www.sciencedirect.com/science/article/pii/S0019057826001461 | degradation stage-aware adversarial DA + Transformer | v26 고급 후보 |
| Joint domain-adaptive Transformer model for bearing RUL prediction across different domains | 2025 | https://www.sciencedirect.com/science/article/pii/S0952197625016173 | domain-adaptive Transformer | TCN/Transformer 이후 DA 결합 후보 |
| Domain adaptation with multi-adversarial learning for open-set cross-domain intelligent bearing fault diagnosis | 2023 | https://ieeexplore.ieee.org/abstract/document/10262196/ | MALDA/MADA류 multi-adversarial DA for bearing diagnosis | RUL 직접 논문은 아니지만 MADA 설명 근거 |
| Failure mechanism information-assisted multi-domain adversarial transfer fault diagnosis model for rolling bearings | 2024 | https://www.mdpi.com/2079-9292/13/11/2133 | DANN/DAN/MADA 비교 및 fault mechanism-assisted DA | MADA 슬라이드 배경 근거 |

## Physics-informed / Trend-residual

| 논문 | 연도 | 링크 | 핵심 | 우리 적용 |
|---|---:|---|---|---|
| Leveraging physics based features for accurate prediction of RUL of bearing | 2025 | https://ieeexplore.ieee.org/abstract/document/11195139 | physics-based features | 물리 기반 피처/보고서 설명 강화 |
| Physics-Informed Transformer for Rolling Bearing RUL Prediction: Dual-Stream Trend-Residual Decomposition | 2026 | https://ieeexplore.ieee.org/abstract/document/11418605 | trend-residual decomposition + Transformer | v26 이후 trend/residual 분리 후보 |
| Physics-informed cross layer temporal frequency transformer network for RUL prediction of rolling bearings | 2025 | https://www.sciencedirect.com/science/article/pii/S0952197625016550 | physics-informed temporal-frequency Transformer | time-frequency + physics-informed 후보 |

## 적용 우선순위

1. v25: 기존 v24에 동역학 피처 추가 (`d_HI`, `slope_HI`, `acc_HI`, `d_env_kurt`, `d_energy_ratio`).
2. v25 sanity check: DTW/phase-space distance로 Test5가 Train EOL 구간과 유사한지 검증.
3. v26: HI-stage pseudo label을 만든 뒤 DANN/MMD/CORAL 또는 MADA류 domain adaptation 실험.
4. v26+: trend-residual decomposition / physics-informed Transformer는 보고서와 고급 실험 후보.
