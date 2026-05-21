# 베어링 진동 데이터 기반 RUL 예측 — 모델링 설계서

> **2026 기계데이터 챌린지** | NSK 30306 테이퍼 롤러 베어링  
> **목표**: Train 베어링(1~4)으로 학습 → Validation 베어링의 RUL(초 단위) 예측

---

## ⚠️ 문제 본질 인식

> 이 문제는 단순한 시계열 회귀가 아니다.  
> **데이터 4개, 결함 모드 4가지, 수명 분산 53%** — 초소량 다도메인 예지보전 문제다.

| 표면적 문제 | 실제 문제 |
|---|---|
| RUL 회귀 | 도메인 이동 + 데이터 희소성 동시 해결 |
| 시계열 예측 | 비정상(Non-stationary) + 변속 환경 적응 |
| RMSE 최소화 | **비대칭 패널티 Score** 직접 최적화 |
| 4개 베어링 학습 | LOBO 검증 없으면 결과 신뢰 불가 |

---

## 0. 데이터 사양 및 핵심 도전

### 0-1. 데이터 사양

```
진동 신호:  25.6 kHz | 4채널 가속도계 (TDMS)
운영 신호:  0.1 Hz   | RPM · Torque · Temp Front/Rear (CSV)
측정 주기:  10분마다 1분 수집 (60s ON / 540s OFF 간헐 측정)
운전 조건:  700~950 RPM (1시간 간격 변경)
베어링:     NSK 30306 Tapered Roller
```

### 0-2. EOL 조건 (먼저 도달하는 것)

```
온도 ≥ 200°C   OR   토크 ≤ −20 Nm   ← 본 데이터셋은 토크 조건 먼저 도달
```

### 0-3. 결함 주파수 (1000 RPM 기준)

| 부위 | 기호 | 주파수 | Order |
|---|---|---|---|
| 내륜 (Inner Race) | BPFI | 140 Hz | 8.40× |
| 외륜 (Outer Race) | BPFO | 93 Hz | 5.58× |
| 볼 (Ball Spin) | BSF | 78 Hz | 4.68× |
| 케이지 (Cage) | FTF | 6.7 Hz | 0.40× |

> **핵심**: `Order = f / (RPM/60)` → RPM 무관 좌표계, 변속 환경에서 필수

### 0-4. Train 베어링별 고장 모드 (분석 결과)

| 베어링 | 수명 | 결함 모드 | 핵심 피처 |
|---|---|---|---|
| Train1 | 21h | 광대역 마모 + 클리어런스 증가 | RMS · std · crest |
| Train2 | 19h | 표면 거칠기 마모 (Stage 3~4) | RMS · std |
| Train3 | 15h | 케이지(FTF) 결함 + 광대역 | FTF 0.40× · shaft 1× |
| Train4 | 23h | 후반 급격 열화 | kurtosis · skew |

> **수명 분산 53%** → 단순 Linear RUL 레이블은 오도(misleading). Lifetime 정규화 필수.

### 0-5. 평가 지표 — 비대칭 패널티 Score

$$
A_{RUL} = \begin{cases}
\exp\!\left(-\ln(0.5) \cdot \dfrac{|Er_i|}{20}\right) & \text{if } Er_i \le 0 \quad \text{(늦은 예측, 강한 패널티)} \\[8pt]
\exp\!\left(+\ln(0.5) \cdot \dfrac{Er_i}{50}\right) & \text{if } Er_i > 0 \quad \text{(이른 예측, 완화)}
\end{cases}
$$

```
Er_i = RUL_pred − RUL_true

늦게 20초 틀리면 → Score × 0.5
일찍 50초 틀리면 → Score × 0.5

즉, 늦은 예측이 2.5배 더 가혹하게 패널티 적용됨
→ 학습 손실함수도 이 비율을 그대로 반영해야 함
```

---

## 1. 신호처리 파이프라인

### 1-1. 전체 흐름

```
TDMS Raw (25.6kHz, 4ch)
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 1. TDMS 병합 & 운영 신호 동기화  │
│  - 126~137개 파일 시간순 concat         │
│  - wf_start_time 기반 CSV 매칭         │
└───────────────┬───────────────────────┘
                │
        ▼
┌───────────────────────────────────────┐
│  STEP 2. Fast Kurtogram               │  ← 학부생과 가장 큰 차이점
│  - 최대 Kurtosis 주파수 대역 자동 탐색  │
│  - 고정 1~6kHz 대신 베어링마다 최적 대역│
│  - 출력: (f_center, bandwidth) per 측정│
└───────────────┬───────────────────────┘
                │
        ▼
┌───────────────────────────────────────┐
│  STEP 3. Angular Resampling           │
│  - Time → Angle 도메인 변환            │
│  - 256 samples/rev                    │
│  - RPM 변동(700~950) 완전 정규화       │
└───────────────┬───────────────────────┘
                │
        ▼
┌───────────────────────────────────────┐
│  STEP 4. VMD (Variational Mode Dec.)  │
│  - K = 4~6 modes                      │
│  - 결함 성분 / 배경 노이즈 분리         │
│  - 초기 미세결함 주파수 돌출 포착       │
└───────────────┬───────────────────────┘
                │
        ▼
┌───────────────────────────────────────┐
│  STEP 5. Hilbert Envelope + 임펄스 검파│
│  e(t) = |x(t) + j·Ĥ(x)(t)|          │
│  - 공진 대역 임펄스 성분 추출          │
└───────────────┬───────────────────────┘
                │
        ▼
┌───────────────────────────────────────┐
│  STEP 6. Order Spectrum               │
│  - FFT in angular domain              │
│  - BPFI/BPFO/BSF/FTF 고조파 추적     │
│  - 결함 에너지 = Σ band(k± 0.15), k=1,2,3 │
└───────────────────────────────────────┘
```

### 1-2. Fast Kurtogram 구현 (핵심)

```python
"""
Antoni & Randall (2006) Fast Kurtogram
: 결함 임펄스 에너지가 집중된 최적 복조 대역을 데이터로부터 자동 탐색

고정 밴드(1~6kHz)보다 우월한 이유:
  - Train3 케이지 결함과 Train1 광대역 마모는 공진 대역이 다름
  - 측정시점(early/late)에 따라 최적 대역이 이동함
"""
from scipy.signal import butter, filtfilt
import numpy as np

def spectral_kurtosis(signal, fs, n_fft=1024, win='hann'):
    """각 주파수 대역의 Kurtosis 계산 → Kurtogram의 기반"""
    from scipy.signal import stft
    f, t, Zxx = stft(signal, fs, window=win, nperseg=n_fft)
    # 각 주파수 빈의 4차 모멘트 / 2차 모멘트² - 2 (kurtosis)
    sk = (np.mean(np.abs(Zxx)**4, axis=1) /
          np.mean(np.abs(Zxx)**2, axis=1)**2) - 2
    return f, sk

def fast_kurtogram(signal, fs=25600, nlevel=8):
    """
    Return: f_center, bandwidth → 이 대역으로 Bandpass 후 Envelope 분석
    """
    best_kurt, best_fc, best_bw = -np.inf, None, None
    
    for level in range(1, nlevel + 1):
        bw = fs / (2 ** (level + 1))
        for k in range(2 ** level):
            fc = bw * (2 * k + 1) / 2
            if fc + bw/2 > fs/2:
                continue
            b, a = butter(4, [max(fc-bw/2, 1)/(fs/2),
                               min(fc+bw/2, fs/2*0.99)/(fs/2)],
                          btype='band')
            filtered = filtfilt(b, a, signal)
            kurt = kurtosis(filtered)  # scipy.stats.kurtosis
            if kurt > best_kurt:
                best_kurt, best_fc, best_bw = kurt, fc, bw
    
    return best_fc, best_bw

# 사용:
# 측정마다 동적으로 최적 대역 탐색 후 해당 대역 Bandpass → Envelope
```

### 1-3. Cyclostationary 지표 (변속 환경 최강)

```python
"""
Cyclostationary Analysis: 
주기적 임펄스를 가진 결함 신호의 통계적 주기성 포착
RPM 변동에 무관하게 동작 → Order Tracking과 상호 보완

IES  (Improved Envelope Spectrum): 노이즈 플로어 제거 후 스펙트럼
CS1  (1st order cyclostationary): 동기 평균 → 결정론적 성분
CS2  (2nd order cyclostationary): 분산 추출 → 랜덤 임펄스 성분 (결함)
"""
def compute_cs2(signal, angular_samples_per_rev=256):
    """2차 Cyclostationary 지표: 결함의 랜덤 임펄스 성분 추출"""
    n_rev = len(signal) // angular_samples_per_rev
    revolutions = signal[:n_rev * angular_samples_per_rev].reshape(n_rev, -1)
    
    # 동기 평균 제거 (CS1 제거) → CS2만 남김
    sync_avg = revolutions.mean(axis=0)
    residual = revolutions - sync_avg
    
    # 분산 스펙트럼 (Variance spectrum)
    cs2 = np.var(residual, axis=0)
    cs2_spectrum = np.abs(np.fft.rfft(cs2))
    return cs2_spectrum
```

---

## 2. Feature Engineering — 31개 피처

### 2-1. 피처 정의표

| 카테고리 | 피처명 | 공식 / 설명 | RUL 관련성 |
|---|---|---|---|
| **시간 도메인** (8) | rms | √(Σx²/N) | 전반적 에너지 증가 |
| | std | σ(x) | 진동 분산 |
| | kurtosis | μ₄/σ⁴ | 충격 임펄스 감도 |
| | crest_factor | peak / rms | 초기 결함 민감 |
| | skewness | μ₃/σ³ | 비대칭 충격 (Train4) |
| | peak | max(|x|) | 최대 충격 크기 |
| | p2p | max(x) - min(x) | 동적 범위 |
| | shape_factor | rms / mean(|x|) | 파형 형태 변화 |
| **Envelope/Order** (8) | env_rms | RMS of envelope | 결함 에너지 총량 |
| | env_kurtosis | kurtosis of envelope | **결함 임펄스 최고 지표** |
| | bpfi_snr | BPFI energy / noise floor | 내륜 결함 강도 |
| | bpfo_snr | BPFO energy / noise floor | 외륜 결함 강도 |
| | bsf_snr | BSF energy / noise floor | 볼 결함 강도 |
| | bpfi_h_ratio | E(1×) / E(2×) + E(3×) | 고조파 발달 정도 |
| | bpfo_h_ratio | — | — |
| | ftf_energy | FTF ± 0.15 order band | Train3 케이지 결함 전용 |
| **Cyclostationary** (4) | ies | Improved Envelope Spectrum | 노이즈 보정 스펙트럼 |
| | cs1 | 동기 평균 에너지 | 결정론적 성분 |
| | cs2 | 분산 스펙트럼 에너지 | **랜덤 임펄스 결함 성분** |
| | alpha | CS2 peak / noise floor | 주기성 집중도 |
| **운영 신호** (6) | torque_std | σ(torque) in window | **최강 선행 지표** |
| | torque_slope | Δtorque / Δt | 토크 추세 |
| | temp_rate | dTemp/dt | 온도 상승 속도 |
| | temp_peak | max(temp) in cycle | 사이클 최고 온도 |
| | rpm_stability | σ(rpm) in cluster | 속도 불안정성 |
| | power_proxy | RPM × \|Torque\| | 마찰열 추정값 |
| **열화 추세** (5) | hi_delta | HI[t] - HI[t-1] | HI 변화율 |
| | hi_accel | hi_delta[t] - hi_delta[t-1] | HI 변화 가속도 |
| | rms_slope5 | OLS slope of RMS (last 5) | 단기 에너지 추세 |
| | temp_slope10 | OLS slope of Temp (last 10) | 중기 온도 추세 |
| | fault_ratio | Σ fault energy / total energy | 결함 에너지 비율 |

### 2-2. 피처 추출 코드 구조

```python
class FeatureExtractor:
    def __init__(self, fs=25600, order_per_rev=256):
        self.fs = fs
        self.opr = order_per_rev
    
    def extract_segment(self, raw_signal, rpm, torque_series, temp_series):
        """1개 측정 세그먼트(60초) → 31개 피처 딕셔너리"""
        
        # 1. Fast Kurtogram → 최적 복조 대역
        fc, bw = fast_kurtogram(raw_signal, self.fs)
        
        # 2. Angular Resampling
        angular_signal = angular_resample(raw_signal, rpm, self.opr)
        
        # 3. VMD 분해 → 결함 성분 선택
        imfs = vmd_decompose(angular_signal, K=5)
        fault_imf = select_fault_imf(imfs)  # Kurtosis 최대 IMF
        
        # 4. Hilbert Envelope
        envelope = np.abs(hilbert(fault_imf))
        order_spectrum = np.abs(np.fft.rfft(envelope))
        
        features = {}
        
        # 시간 도메인
        features.update(self._time_domain(raw_signal))
        
        # Envelope / Order 도메인
        features.update(self._envelope_features(envelope, order_spectrum, rpm))
        
        # Cyclostationary
        features.update(self._cyclostationary(angular_signal))
        
        # 운영 신호
        features.update(self._operational(torque_series, temp_series, rpm))
        
        return features
    
    def _time_domain(self, x):
        from scipy.stats import kurtosis, skew
        return {
            'rms': np.sqrt(np.mean(x**2)),
            'std': np.std(x),
            'kurtosis': kurtosis(x),
            'crest_factor': np.max(np.abs(x)) / np.sqrt(np.mean(x**2)),
            'skewness': skew(x),
            'peak': np.max(np.abs(x)),
            'p2p': np.ptp(x),
            'shape_factor': np.sqrt(np.mean(x**2)) / np.mean(np.abs(x))
        }
```

---

## 3. Health Indicator — DTC-VAE

> **핵심 개념**: 수동 가중 합산(학부생 방식) 대신,  
> VAE가 단조성(Monotonicity) 제약을 내재화하여 **데이터로부터 HI를 자동 학습**

### 3-1. 왜 HI가 필요한가

```
Raw 피처 31개를 직접 RUL 모델에 넣으면:
  - 노이즈에 민감
  - 베어링마다 다른 스케일
  - 열화 방향성(단조성) 보장 없음
  → 회귀 모델 수렴 불안정

HI = 단조 감소하는 1차원 열화 상태 지표
  - 단조성 → 모델이 "감소 방향"을 쉽게 학습
  - 1차원 압축 → 노이즈 억제, 일반화 향상
  - 해석 가능 → 물리적 의미 존재
```

### 3-2. DTC-VAE 아키텍처

```
참고 논문:
  Hybrid CAE with monotonicity constraint
  Engineering Applications of Artificial Intelligence, 2025
  → RUL 예측 오차 85% 감소 보고

  Unsupervised HI via multi-criterion selection + Attentive VAE
  Science China Technological Sciences, 2024
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

class DTC_VAE(nn.Module):
    """
    Degradation-Trend-Constrained Variational Autoencoder
    
    Loss = L_recon + β·L_kl + λ₁·L_monotonic + λ₂·L_trend
    
    L_monotonic : 시간 순서대로 z가 증가(열화)하도록 강제
    L_trend     : z와 시간 인덱스 간 Spearman 상관 최대화
    """
    def __init__(self, input_dim=31, hidden_dim=64, latent_dim=1):
        super().__init__()
        
        # Encoder: 피처 → μ, log_σ² (잠재 변수)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 32),         nn.LayerNorm(32),         nn.GELU(),
        )
        self.fc_mu    = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder: z → 피처 복원
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),      nn.LayerNorm(32),         nn.GELU(),
            nn.Linear(32, hidden_dim),      nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x):
        h    = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z    = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z
    
    def loss(self, x_seq, beta=0.1, lambda1=1.0, lambda2=0.5):
        """
        x_seq: [T, input_dim]  (T = 측정 횟수, 시간 순서)
        """
        x_hat, mu, logvar, z = self.forward(x_seq)
        T = x_seq.shape[0]
        
        # 1. 재구성 손실
        L_recon = F.mse_loss(x_hat, x_seq)
        
        # 2. KL 발산
        L_kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        
        # 3. 단조성 손실 (핵심)
        # z[t+1] >= z[t] 이어야 함 (열화 = 증가)
        # 위반 시 패널티: max(0, z[t] - z[t+1])
        diffs = z[1:] - z[:-1]          # [T-1, 1]
        L_mono = torch.mean(F.relu(-diffs))  # 감소하면 패널티
        
        # 4. 추세성 손실 (z와 시간의 상관 최대화)
        time_idx = torch.arange(T, dtype=torch.float32, device=x_seq.device)
        z_flat = z.squeeze()
        # 정규화 후 내적 → 선형 상관 근사
        z_norm = (z_flat - z_flat.mean()) / (z_flat.std() + 1e-8)
        t_norm = (time_idx - time_idx.mean()) / (time_idx.std() + 1e-8)
        L_trend = 1.0 - (z_norm * t_norm).mean()
        
        total = L_recon + beta * L_kl + lambda1 * L_mono + lambda2 * L_trend
        return total, {
            'recon': L_recon.item(), 'kl': L_kl.item(),
            'mono': L_mono.item(),   'trend': L_trend.item()
        }

# HI 품질 평가 (3가지 기준)
def evaluate_hi_quality(z_sequence):
    T = len(z_sequence)
    time_idx = np.arange(T)
    
    # 단조성: 증가 횟수 비율
    monotonicity = np.mean(np.diff(z_sequence) > 0)
    
    # 추세성: Spearman 상관
    trendability, _ = spearmanr(z_sequence, time_idx)
    
    # 강건성: 1 - (std / range)
    robustness = 1 - (np.std(np.diff(z_sequence)) /
                      (np.max(z_sequence) - np.min(z_sequence) + 1e-8))
    
    total_score = (monotonicity + trendability + robustness) / 3
    return {'monotonicity': monotonicity, 'trendability': trendability,
            'robustness': robustness, 'total': total_score}
```

---

## 4. RUL 레이블링 — CUSUM 기반 FPT 자동 탐지

### 4-1. First Prediction Time (FPT) 탐지

```python
"""
학부생: 수동 임계값으로 FPT 설정
교수:   CUSUM (누적합 관리도) → 통계적 변화점 자동 탐지

CUSUM: 정상 분포 기반, 시그널이 μ에서 벗어나기 시작하는 시점 탐지
       산업 표준 통계적 공정 관리(SPC) 기법
"""
def detect_fpt_cusum(hi_sequence, k=0.5, h=5.0, init_ratio=0.2):
    """
    hi_sequence: HI 시계열 (길이 T)
    k: slack parameter (drift 허용 수준)
    h: threshold (알람 임계값)
    init_ratio: 정상 구간 추정에 사용할 초기 데이터 비율
    
    Return: FPT 인덱스
    """
    n_init = max(5, int(len(hi_sequence) * init_ratio))
    baseline = hi_sequence[:n_init]
    mu0 = baseline.mean()
    sigma0 = baseline.std() + 1e-8
    
    S_pos = 0.0  # 상향 이탈 누적합
    
    for i in range(n_init, len(hi_sequence)):
        xi = (hi_sequence[i] - mu0) / sigma0
        S_pos = max(0, S_pos + xi - k)
        
        if S_pos > h:
            return i  # FPT 탐지
    
    # 탐지 못했으면 마지막 20% 시점으로 보수적 설정
    return int(len(hi_sequence) * 0.8)
```

### 4-2. Piecewise Linear RUL 레이블

```python
def generate_rul_labels(n_measurements, T_f_seconds, T_fpt_idx,
                        measurement_timestamps):
    """
    n_measurements: 총 측정 횟수
    T_f_seconds:    EOL까지 남은 시간 (초)
    T_fpt_idx:      FPT 탐지 인덱스
    measurement_timestamps: 각 측정의 절대 시간 (초)
    
    Return: RUL 레이블 배열 (초 단위)
    """
    rul_at_fpt = T_f_seconds - measurement_timestamps[T_fpt_idx]
    labels = []
    
    for i, t in enumerate(measurement_timestamps):
        if i < T_fpt_idx:
            labels.append(rul_at_fpt)  # 정상 구간: 고정 최대값
        else:
            labels.append(max(0, T_f_seconds - t))  # 열화 구간: 선형 감소
    
    return np.array(labels)

# 시각화 기대 형태:
#
# RUL
#  ↑
#  |‾‾‾‾‾‾‾‾‾\
#  |  고정     \  선형 감소
#  |            \
#  |             \
#  └──────────────→ 측정 인덱스
#        FPT↑
```

---

## 5. 모델 설계 — 3계층 앙상블

> **핵심 원칙**: 데이터 4개 베어링 → 단일 거대 모델 = 과적합 필연  
> 서로 다른 귀납 편향을 가진 모델을 앙상블하는 것이 최적

### 5-1. 전체 앙상블 구조

```
                         [입력]
               31개 피처 + HI  (측정 인덱스 × 32차원)
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌───────────┐    ┌────────────┐    ┌───────────┐
    │ Layer 1   │    │  Layer 2   │    │  Layer 3  │
    │ XGBoost   │    │  TCN-TFT   │    │    GPR    │
    │ (tabular) │    │ (temporal) │    │ (Bayes)   │
    └─────┬─────┘    └─────┬──────┘    └─────┬─────┘
          │                │                 │
          └────────────────┼─────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Meta-Learner   │
                  │  (Ridge + Asym  │
                  │   Loss)         │
                  └────────┬────────┘
                           │
                           ▼
                  RUL 예측 + 불확실성 구간
```

### 5-2. Layer 1: XGBoost with 비대칭 손실

```python
import xgboost as xgb
import numpy as np

def asymmetric_gradient_hessian(predt: np.ndarray, dtrain: xgb.DMatrix):
    """
    챌린지 평가 지표 A(RUL)를 직접 미분하여 XGBoost 손실함수로 구현
    
    A(Er) = exp(-ln(0.5) · |Er| / 20)  if Er ≤ 0  (늦은 예측)
    A(Er) = exp( ln(0.5) ·  Er  / 50)  if Er > 0  (이른 예측)
    
    손실 L = 1 - A(Er)  →  최소화
    """
    labels = dtrain.get_label()
    error  = predt - labels       # Er = pred - true
    
    ln_half = np.log(0.5)         # ≈ -0.693
    
    # 1차 미분 (gradient)
    grad = np.where(
        error <= 0,
        # 늦은 예측: dL/dEr = ln(0.5)/20 · exp(-ln(0.5)·|Er|/20)
        (ln_half / 20) * np.exp(-ln_half * (-error) / 20),
        # 이른 예측: dL/dEr = ln(0.5)/50 · exp( ln(0.5)· Er/50)
        (ln_half / 50) * np.exp( ln_half *   error  / 50)
    )
    
    # 2차 미분 (hessian)
    hess = np.where(
        error <= 0,
        (ln_half / 20)**2 * np.exp(-ln_half * (-error) / 20),
        (ln_half / 50)**2 * np.exp( ln_half *   error  / 50)
    )
    
    return grad, hess

def asymmetric_eval(predt, dtrain):
    """XGBoost eval_metric용"""
    labels = dtrain.get_label()
    error  = predt - labels
    score  = np.where(
        error <= 0,
        np.exp(-np.log(0.5) * (-error) / 20),
        np.exp( np.log(0.5) *   error  / 50)
    )
    return 'asym_score', float(score.mean())

# 학습
params = {
    'max_depth': 5,
    'learning_rate': 0.02,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': 42
}

model_xgb = xgb.train(
    params,
    dtrain,
    obj=asymmetric_gradient_hessian,
    feval=asymmetric_eval,
    num_boost_round=1000,
    early_stopping_rounds=50,
    evals=[(dval, 'val')],
    maximize=True  # score 최대화
)
```

### 5-3. Layer 2: TCN + Temporal Fusion Transformer

```
참고 논문:
  "Temporal convolutional and fusional transformer model with Bi-LSTM 
   encoder-decoder for multi-time-window RUL prediction"
  arXiv:2511.04723, 2024
  → 기존 SOTA 대비 RMSE 5.5% 추가 감소

아키텍처 흐름:
  [측정 시퀀스]
        │
   TCN Encoder       ← 국소 패턴 (결함 주기, 단기 추세)
        │
   TFT Encoder       ← 전역 열화 추세 + 변수 중요도 해석
        │
   Bi-LSTM Bridge    ← 양방향 시퀀스 압축
        │
   Quantile Heads    ← q10, q50, q90 (불확실성 정량화)
        │
   RUL Output (초 단위)
```

```python
import torch
import torch.nn as nn

class TemporalConvBlock(nn.Module):
    """TCN: 인과적 확장 합성곱으로 국소 시간 패턴 추출"""
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv  = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.chomp = nn.Identity()   # causal: 오른쪽 pad 제거
        self.norm  = nn.LayerNorm(out_ch)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(0.2)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)[..., :x.shape[-1]]   # causal chomp
        out = self.drop(self.act(
              self.norm(out.transpose(1,2)).transpose(1,2)
        ))
        return out + self.res(x)


class TCN_TFT_RUL(nn.Module):
    """
    TCN + Temporal Fusion Transformer 하이브리드
    
    TFT 핵심 컴포넌트:
      - Variable Selection Network (VSN): 측정마다 관련 피처 자동 선택
      - Gated Residual Network (GRN):    불필요 피처 억제
      - Multi-head Attention:             장기 열화 추세 포착
      - Quantile Output:                  예측 불확실성 정량화
    """
    def __init__(self, input_dim=32, d_model=64, n_heads=4,
                 seq_len=20, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
        # TCN: 국소 패턴 (dilations: 1,2,4,8)
        self.tcn = nn.Sequential(
            TemporalConvBlock(input_dim, 64, dilation=1),
            TemporalConvBlock(64,        64, dilation=2),
            TemporalConvBlock(64,        64, dilation=4),
            TemporalConvBlock(64,        64, dilation=8),
        )
        
        # Variable Selection Network
        self.vsn_weight = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # TFT Encoder (Multi-head Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=256, dropout=0.2,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Bi-LSTM Bridge
        self.bilstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.2)
        
        # 정적 공변량 (베어링 ID, RPM 클러스터)
        self.static_emb = nn.Embedding(10, d_model)
        
        # Quantile 출력 헤드
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in quantiles
        ])
    
    def forward(self, x_seq, static_ids):
        """
        x_seq:     [B, T, input_dim]
        static_ids: [B]  (베어링 ID 등 정적 정보)
        """
        B, T, D = x_seq.shape
        
        # Variable Selection
        weights = self.vsn_weight(x_seq)      # [B, T, D]
        x_sel   = x_seq * weights
        
        # TCN: 국소 특징
        tcn_out = self.tcn(x_sel.transpose(1,2)).transpose(1,2)  # [B, T, 64]
        
        # 정적 공변량 주입
        static_feat = self.static_emb(static_ids).unsqueeze(1)   # [B, 1, d_model]
        
        # Transformer: 전역 추세
        # (차원 맞추기용 projection 생략, 실제 구현 시 추가)
        tft_out = self.transformer(tcn_out + static_feat)         # [B, T, 64]
        
        # Bi-LSTM: 시퀀스 압축
        lstm_out, _ = self.bilstm(tft_out)     # [B, T, 64]
        context = lstm_out[:, -1, :]           # 마지막 시점 컨텍스트 [B, 64]
        
        # Quantile 예측
        preds = [head(context) for head in self.output_heads]
        return torch.cat(preds, dim=-1)        # [B, n_quantiles]

# 학습: q50이 주 예측, q10/q90으로 불확실성 구간 제공
# 비대칭 패널티 고려: q50 보다 약간 작은 값(q40 수준) 제출 권장
```

### 5-4. Layer 3: Gaussian Process Regression (불확실성 특화)

```python
"""
GPR의 강점:
  - 단 4개 베어링으로도 통계적 신뢰 구간 제공
  - 커널 설계로 열화 물리를 사전 지식으로 주입 가능
  - 소량 데이터에서 딥러닝보다 안정적
  
열화 전용 커널 설계:
  Matern(ν=2.5)  : 한 번 미분 가능한 매끄러운 열화 곡선
  RBF            : 장기 상관관계 (수명 전반의 추세)
  WhiteKernel    : 측정 노이즈 모델링
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel as C
)

kernel = (
    C(1.0, (1e-3, 1e3)) *
    Matern(length_scale=10.0, nu=2.5) +
    C(0.5, (1e-3, 1e3)) *
    RBF(length_scale=50.0) +
    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,
    alpha=1e-2,
    normalize_y=True
)
gpr.fit(X_train, y_train)

# 예측: μ ± σ
rul_mu, rul_sigma = gpr.predict(X_test, return_std=True)

# 비대칭 패널티 전략:
#   늦은 예측이 2.5배 더 패널티 → 약간 이른 예측이 유리
#   → μ - 0.3σ 제출 (보수적 하한)
rul_conservative = rul_mu - 0.3 * rul_sigma
```

### 5-5. Meta-Learner (Stacking)

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class AsymmetricMetaLearner:
    """
    XGBoost, TCN-TFT, GPR의 예측을 입력으로 받아
    최종 RUL 예측 (비대칭 손실 최적화)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.ridge  = Ridge(alpha=1.0)
    
    def fit(self, preds_xgb, preds_tft_q50, preds_gpr_mu, preds_gpr_sigma, y_true):
        # 스택 피처 구성
        X_meta = np.column_stack([
            preds_xgb,
            preds_tft_q50,
            preds_gpr_mu,
            preds_gpr_sigma,      # 불확실성 정보도 활용
            preds_gpr_mu - 0.3 * preds_gpr_sigma   # 보수적 예측
        ])
        X_scaled = self.scaler.fit_transform(X_meta)
        
        # 비대칭 가중 최적화
        # 늦은 오차(er < 0)에 더 높은 가중치 적용
        sample_weight = np.where(
            (preds_gpr_mu - y_true) < 0,   # 늦은 예측
            2.5,                             # 2.5배 가중치
            1.0                              # 이른 예측
        )
        self.ridge.fit(X_scaled, y_true, sample_weight=sample_weight)
    
    def predict(self, *preds_tuple):
        X_meta = np.column_stack(list(preds_tuple))
        X_scaled = self.scaler.transform(X_meta)
        return self.ridge.predict(X_scaled)
```

---

## 6. 도메인 적응 — DANN (베어링 간 일반화)

```
문제: Train1~4는 각자 다른 결함 모드 = 다른 도메인
     Validation 베어링 = 본 적 없는 분포

해결: Domain-Adversarial Neural Network (DANN)
     "결함 진단에 유용하되, 어떤 베어링인지 구분 못하는" 표현 학습
     
참고: CNN-Bi-LSTM Domain Adaptation for RUL, Sensors 2024
      PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11548141/
```

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None   # gradient 방향 반전

class DANN_RUL(nn.Module):
    def __init__(self, input_dim=32, d_model=64, n_bearings=4):
        super().__init__()
        
        # 공유 특징 추출기
        self.feature_extractor = TCN_TFT_RUL(input_dim, d_model)
        
        # RUL 예측 헤드 (정방향)
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # 도메인 분류 헤드 (역전파 시 gradient 반전)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, n_bearings)
        )
    
    def forward(self, x_seq, static_ids, alpha=1.0):
        features = self.feature_extractor(x_seq, static_ids)   # [B, d_model]
        
        # RUL 예측
        rul_pred = self.rul_head(features)
        
        # 도메인 예측 (GRL로 gradient 반전)
        reversed_feat = GradientReversalFunction.apply(features, alpha)
        domain_pred   = self.domain_head(reversed_feat)
        
        return rul_pred, domain_pred

# 학습
rul_loss    = asymmetric_rul_loss(rul_pred, rul_true)
domain_loss = F.cross_entropy(domain_pred, bearing_id_labels)

# alpha: 학습 진행에 따라 0→1 증가 (GRL 강도 점진적 증가)
alpha = 2 / (1 + np.exp(-10 * progress)) - 1
total_loss = rul_loss + 0.1 * alpha * domain_loss
```

---

## 7. Self-Supervised Pre-training (데이터 부족 극복)

```
아이디어:
  레이블(RUL) 없이도 "어떤 시간대 측정인지"는 알 수 있음
  → 인접 시간대 = 비슷한 열화 상태 = Positive pair
  → 먼 시간대  = 다른 열화 상태 = Negative pair
  → Contrastive Learning으로 열화 표현 자동 학습

참고:
  Dual-dimensional Contrastive SSL for Rolling Bearing RUL
  Nature Scientific Reports, 2025
  https://www.nature.com/articles/s41598-026-38417-7
  
  Enhancing prognostics for sparse labeled data via SSL
  Engineering Applications of AI, 2024
```

```python
class DualDimContrastiveSSL(nn.Module):
    """
    두 가지 대조 학습 동시 수행:
    
    1. Temporal-level Contrastive:
       같은 베어링의 인접 측정 → 유사해야 (Positive)
       먼 시간 측정 → 달라야 (Negative)
    
    2. Instance-level Contrastive:
       다른 베어링이라도 HI가 비슷한 시점 → 유사해야 (Positive)
       HI가 크게 다른 시점 → 달라야 (Negative)
    """
    def __init__(self, encoder, proj_dim=64, temperature=0.1):
        super().__init__()
        self.encoder    = encoder
        self.projector  = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, 32)
        )
        self.temp = temperature
    
    def temporal_contrastive_loss(self, z_seq, time_diffs):
        """
        z_seq:      [T, proj_dim]
        time_diffs: T×T 행렬, 시간 차이 (측정 인덱스 단위)
        
        가까운 시간 → Positive, 먼 시간 → Negative
        """
        z = F.normalize(self.projector(z_seq), dim=-1)
        sim = torch.mm(z, z.T) / self.temp                # [T, T]
        
        # 시간 차이가 작을수록 Positive
        pos_mask = (time_diffs < 3).float()               # 3 측정 이내 = Positive
        neg_mask = (time_diffs > 10).float()              # 10 측정 이상 = Negative
        
        loss = -torch.log(
            torch.exp(sim * pos_mask).sum(-1) /
            (torch.exp(sim * neg_mask).sum(-1) + 1e-8)
        ).mean()
        return loss
    
    def instance_contrastive_loss(self, z_batch, hi_batch):
        """HI 값이 비슷한 측정끼리 유사한 표현을 갖도록"""
        z = F.normalize(self.projector(z_batch), dim=-1)
        sim = torch.mm(z, z.T) / self.temp
        
        hi_diffs = torch.cdist(hi_batch.unsqueeze(-1),
                               hi_batch.unsqueeze(-1)).squeeze()
        pos_mask = (hi_diffs < 0.1).float()   # HI 차이 10% 이내 = Positive
        
        loss = -torch.log(
            torch.exp(sim * pos_mask).sum(-1) /
            torch.exp(sim).sum(-1)
        ).mean()
        return loss

# 학습 순서:
# 1단계: Self-supervised pre-training (레이블 없이, 모든 측정 활용)
# 2단계: RUL regression fine-tuning (레이블 있는 것만, 작은 LR)
```

---

## 8. 학습 전략

### 8-1. 검증 방법 — Leave-One-Bearing-Out (LOBO)

```
⛔ 절대 금지: 측정 단위 무작위 split
             (같은 베어링의 앞/뒤 측정이 train/val로 갈리면 → 정보 누출)

✅ 필수: Leave-One-Bearing-Out Cross-Validation

Fold 1: Train2,3,4 → 학습 / Train1 → 검증
Fold 2: Train1,3,4 → 학습 / Train2 → 검증
Fold 3: Train1,2,4 → 학습 / Train3 → 검증
Fold 4: Train1,2,3 → 학습 / Train4 → 검증

→ 4개 Fold의 Meta-Learner 앙상블로 최종 Validation 예측
→ 각 Fold에서 비대칭 Score를 주 지표로 모니터링
```

### 8-2. 학습 파라미터 권장값

```python
training_config = {
    # Pre-training (SSL)
    "ssl": {
        "epochs": 200,
        "lr": 3e-4,
        "batch_size": 32,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
    },
    
    # Fine-tuning (RUL)
    "finetune": {
        "epochs": 500,
        "lr": 1e-4,           # Pre-training보다 작게
        "batch_size": 16,     # 소량 데이터 → 작은 배치
        "optimizer": "AdamW",
        "weight_decay": 1e-3, # 강한 정규화
        "dropout": 0.3,       # 소량 데이터 → 강한 Dropout
        "label_smoothing": 0.05,
        "scheduler": "CosineAnnealingWarmRestarts",
        "T_0": 100,
        "early_stopping_patience": 50,
        "early_stopping_metric": "asym_score",  # ← RMSE 아님!
    },
    
    # 데이터 증강 비율
    "augmentation": {
        "noise_injection_snr_range": (5, 20),   # dB
        "time_warp_rate": 0.1,                  # ±10%
        "gan_ratio": 2.0,                        # 고장 근접 구간 2배 증강
        "target_rul_threshold_pct": 0.2,        # RUL < 20% 구간 집중 증강
    }
}
```

### 8-3. Early Stopping — 비대칭 Score 기준

```python
class AsymmetricEarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
    
    def compute_score(self, rul_pred, rul_true):
        error = rul_pred - rul_true
        score = np.where(
            error <= 0,
            np.exp(-np.log(0.5) * (-error) / 20),
            np.exp( np.log(0.5) *   error  / 50)
        )
        return score.mean()
    
    def step(self, rul_pred, rul_true):
        score = self.compute_score(rul_pred, rul_true)
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return False   # 계속 학습
        else:
            self.counter += 1
            return self.counter >= self.patience   # True = 중단
```

---

## 9. 데이터 증강

### 9-1. 전략 요약

| 기법 | 구현 | 목적 |
|---|---|---|
| Noise Injection | SNR 5~20dB 가우시안 노이즈 | 전기적 노이즈 환경 강건성 |
| Time Warping | ±10% 측정 인덱스 보간 | 가변 RPM 강건성 |
| Magnitude Scaling | 0.8~1.2× 진폭 스케일 | 센서 교정 오차 모사 |
| **GAN 합성** | **조건부 GAN (고장 근접 특화)** | **RUL<20% 클래스 불균형 보정** |
| Window Slicing | 60초 내 랜덤 슬라이딩 윈도우 | 세그먼트 수 증가 |

### 9-2. GAN 데이터 증강 (고장 근접 구간 집중)

```python
"""
전략:
  - 정상 구간 (~90%): 증강 불필요
  - 고장 근접 (RUL < 20%): GAN으로 합성 데이터 생성 → 클래스 불균형 보정
  - Sobol Sampling: 물리적 타당성 확보 (균일 분포 샘플링)

주의:
  - GAN 합성 데이터는 Train에만 사용
  - Validation / Test 세트에는 절대 포함 금지 (Leakage)
"""
class ConditionalWGAN_GP(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty
    조건: 현재 HI 값 (열화 단계 조건부 생성)
    """
    def __init__(self, feature_dim=31, condition_dim=1, latent_dim=32):
        super().__init__()
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim + condition_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    
    def generate(self, hi_condition, n_samples):
        """HI 조건부 합성 피처 생성"""
        z = torch.randn(n_samples, 32)
        cond = hi_condition.expand(n_samples, -1)
        return self.generator(torch.cat([z, cond], dim=-1))
```

---

## 10. 구현 로드맵

### Phase 0: 환경 구성 (1일)

```bash
pip install nptdms scipy scikit-learn xgboost torch torchvision
pip install pytorch-forecasting gpytorch pywavelets
```

```
필요 데이터:
  Train1~4: TDMS 파일 + CSV 파일
  파일 구조: 
    data/
      train1/
        vibration_*.tdms
        operating_*.csv
      train2/ ...
```

### Phase 1: 신호처리 + 피처 추출 (3~4일)

```
Day 1: TDMS 읽기 + 운영 신호 동기화
Day 2: Fast Kurtogram + Angular Resampling 구현
Day 3: VMD + Envelope + Order Spectrum
Day 4: 31개 피처 추출 → features_bearingN.csv 저장
```

### Phase 2: HI 구성 + 레이블링 (2~3일)

```
Day 5: DTC-VAE 학습 (4개 베어링 동시)
Day 6: CUSUM FPT 탐지 + Piecewise RUL 레이블 생성
Day 7: HI 품질 평가 (단조성·추세성·강건성)
```

### Phase 3: 베이스라인 모델 (2~3일)

```
Day 8:  XGBoost + 비대칭 손실 → 빠른 성능 확인
Day 9:  LOBO Cross-Validation 세팅 + 검증
Day 10: GPR 모델 학습 + 불확실성 구간 확인
```

### Phase 4: SOTA 모델 (4~5일)

```
Day 11~12: SSL Pre-training (Dual-dim Contrastive)
Day 13~14: TCN-TFT Fine-tuning + DANN 적용
Day 15:    Meta-Learner Stacking
```

### Phase 5: 최적화 + 제출 (2~3일)

```
Day 16: 비대칭 Early Stopping 튜닝
Day 17: Ensemble 가중치 최적화 (비대칭 Score 기준)
Day 18: 최종 검증 + Validation 제출
```

---

## 11. 핵심 차별점 요약

| 항목 | 일반적 접근 | 본 설계 (SOTA) |
|---|---|---|
| **복조 대역** | 고정 1~6kHz | Fast Kurtogram 적응형 탐색 |
| **HI 구성** | 수동 가중 합산 | DTC-VAE 단조성 제약 자동 학습 |
| **FPT 탐지** | 수동 임계값 | CUSUM 통계적 변화점 자동 탐지 |
| **검증 방법** | 측정 단위 random split | LOBO Cross-Validation |
| **손실 함수** | MSE / Huber | 비대칭 지수 손실 (평가 지표 직접 구현) |
| **일반화** | 없음 | DANN 도메인 적응 |
| **불확실성** | 점 추정 | GPR + Quantile TFT (신뢰 구간) |
| **데이터 부족** | GAN 증강만 | SSL Pre-training + GAN 앙상블 |
| **모델** | 단일 CNN-LSTM | XGBoost + TCN-TFT + GPR Stacking |

---

## 참고 문헌

| 논문 | 저널 | 핵심 기여 |
|---|---|---|
| [Physics-informed multi-state temporal frequency network](https://ideas.repec.org/a/eee/reensy/v242y2024ics0951832023006300.html) | RE&SS 2024 | 물리 기반 다중 상태 주파수 네트워크 |
| [Dual-dimensional Contrastive SSL for RUL](https://www.nature.com/articles/s41598-026-38417-7) | Nature Sci.Reports 2025 | 이중 차원 대조 자기지도 학습 |
| [TCN-TFT Bi-LSTM for RUL](https://arxiv.org/pdf/2511.04723) | arXiv 2024 | RMSE 5.5% 추가 감소 |
| [Hybrid CAE monotonic HI](https://www.sciencedirect.com/science/article/pii/S095219762401635X) | Eng.App.AI 2025 | RUL 오차 85% 감소 |
| [CNN-Bi-LSTM Domain Adaptation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548141/) | Sensors 2024 | 베어링 간 도메인 이동 극복 |
| [Asymmetric Loss for RUL](https://ieeexplore.ieee.org/iel7/9200848/9206590/09207051.pdf) | IEEE 2020 | 비대칭 손실함수 이론 기반 |
| [Attentive VAE + Multi-criterion HI](https://link.springer.com/article/10.1007/s11431-023-2610-4) | Science China 2024 | 다기준 HI 자동 구성 |
| Fast Kurtogram (Antoni & Randall) | Mech.Syst.Sig.Proc. 2006 | 최적 복조 대역 자동 탐색 |
| [CUSUM for bearing FPT detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6263687/) | Sensors 2018 | 통계적 변화점 탐지 |

---

*본 설계서는 2026 기계데이터 챌린지를 위한 세계 최고 수준의 RUL 예측 프레임워크이다.*  
*구현 시 LOBO 검증과 비대칭 손실함수 적용을 반드시 준수할 것.*
