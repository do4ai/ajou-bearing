"""
합성 베어링 진동 데이터 생성기
NSK 30306 테이퍼 롤러 베어링 스펙 기반
챌린지 데이터와 동일한 특성 시뮬레이션

Bearing specs:
  Fs = 25600 Hz  (25.6 kHz)
  BPFI = 8.40x  (inner race fault)
  BPFO = 5.58x  (outer race fault)
  BSF  = 4.68x  (ball spin fault)
  FTF  = 0.40x  (cage fault)

Train 베어링별 고장 모드:
  Train1: 광대역 마모 (RMS·std·crest 주도)     수명 ~21h
  Train2: 표면 거칠기 마모                       수명 ~19h
  Train3: 케이지(FTF) 결함                       수명 ~15h
  Train4: 후반 급격 열화 (kurtosis·skew 급변)   수명 ~23h
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

FS = 25600          # 샘플링 주파수
SEG_DUR = 2         # 저장 세그먼트 길이 (초) — 실제 60초에서 대표 2초 추출
INTERVAL = 600      # 측정 간격 (초, 10분)
MEAS_DUR = 2        # 1회 저장 길이 (초) — 메모리 절약용, 특징 추출 충분

# 결함 Order (RPM 무관)
ORDERS = {'BPFI': 8.40, 'BPFO': 5.58, 'BSF': 4.68, 'FTF': 0.40}

# EOL 조건 (토크 -20Nm 초과)
EOL_TORQUE = -20.0


def rpm_schedule(t_hours, base_rpm=800, pattern='alternating'):
    """RPM 스케줄: 700~950 1시간 단위 변경"""
    rpm_levels = [700, 800, 900, 950, 750, 850]
    idx = int(t_hours) % len(rpm_levels)
    return rpm_levels[idx]


def generate_bearing_signal(t_seconds, rpm, health_ratio, fault_mode,
                             n_samples=None, noise_snr=15):
    """
    단일 세그먼트 진동 신호 생성

    Parameters:
        t_seconds   : 현재 시간 (초, 절대)
        rpm         : 현재 RPM
        health_ratio: 0(완전 정상) ~ 1(EOL 직전)
        fault_mode  : 'wideband'|'surface'|'cage'|'sudden'
        n_samples   : 샘플 수 (None이면 FS*MEAS_DUR)
        noise_snr   : 신호대 잡음비 (dB)
    Returns:
        signal: np.ndarray [n_samples]
    """
    if n_samples is None:
        n_samples = FS * MEAS_DUR

    t = np.linspace(0, MEAS_DUR, n_samples, endpoint=False)
    fr = rpm / 60.0   # 회전 주파수 (Hz)

    # ─── 1. 정상 배경 진동 ───
    # 축 회전 + 조화파
    signal = 0.05 * np.sin(2 * np.pi * fr * t)
    signal += 0.02 * np.sin(2 * np.pi * 2 * fr * t)

    # ─── 2. 환경 노이즈 (전원 60Hz 고조파 + 모터 슬롯 ~160Hz) ───
    signal += 0.03 * np.sin(2 * np.pi * 120 * t)
    signal += 0.02 * np.sin(2 * np.pi * 240 * t)
    signal += 0.015 * np.sin(2 * np.pi * 160 * t)

    # ─── 3. 결함 임펄스 생성 ───
    fault_amp = health_ratio ** 1.5   # 열화 진행에 따라 비선형 증가

    if fault_mode == 'wideband':
        # Train1: 광대역 마모 → RMS/std 전반적 증가
        f_fault = ORDERS['BPFO'] * fr
        for k in [1, 2, 3]:
            signal += fault_amp * (0.3 / k) * np.sin(2 * np.pi * k * f_fault * t)
        # 광대역 에너지 증가
        signal += fault_amp * 0.2 * np.random.randn(n_samples)

    elif fault_mode == 'surface':
        # Train2: 표면 거칠기 → RMS + 노이즈 플로어 증가
        f_fault = ORDERS['BPFI'] * fr
        for k in [1, 2, 3]:
            signal += fault_amp * (0.25 / k) * np.sin(2 * np.pi * k * f_fault * t)
        signal += fault_amp * 0.3 * np.random.randn(n_samples)

    elif fault_mode == 'cage':
        # Train3: 케이지(FTF) 결함 → FTF 라인 발달
        f_ftf = ORDERS['FTF'] * fr
        f_shaft = fr
        for k in [1, 2, 3]:
            signal += fault_amp * (0.4 / k) * np.sin(2 * np.pi * k * f_ftf * t)
        signal += fault_amp * 0.35 * np.sin(2 * np.pi * f_shaft * t)
        # 임펄스 트레인 (케이지 결함 특성)
        impulse_period = int(FS / f_ftf)
        impulse_times = np.arange(0, n_samples, impulse_period)
        for imp_t in impulse_times:
            if imp_t + 100 < n_samples:
                decay = np.exp(-np.arange(100) / 20)
                signal[imp_t:imp_t+100] += fault_amp * 0.5 * decay

    elif fault_mode == 'sudden':
        # Train4: 후반 급격 열화 → kurtosis/skew 급변
        f_fault = ORDERS['BPFI'] * fr
        for k in [1, 2, 3]:
            signal += fault_amp * 0.2 * np.sin(2 * np.pi * k * f_fault * t)
        # 후반부 급격 임펄스 (비선형 가속)
        if health_ratio > 0.7:
            sudden_factor = (health_ratio - 0.7) / 0.3 * 3
            impulse_period = int(FS / (ORDERS['BPFO'] * fr))
            impulse_times = np.arange(0, n_samples, impulse_period)
            for imp_t in impulse_times:
                if imp_t + 50 < n_samples:
                    decay = np.exp(-np.arange(50) / 10)
                    signal[imp_t:imp_t+50] += sudden_factor * fault_amp * decay

    # ─── 4. 가우시안 측정 노이즈 ───
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (noise_snr / 10))
    signal += np.sqrt(noise_power) * np.random.randn(n_samples)

    return signal.astype(np.float32)


def simulate_temperature(t_hours, health_ratio, fault_mode):
    """온도 시뮬레이션 (Front/Rear 구분)"""
    base_temp = 25.0
    operating_temp = 80 + 60 * (t_hours / 24)   # 시간에 따라 점진 증가
    fault_heat = health_ratio ** 2 * 120         # 결함에 따른 발열

    # 사이클 패턴 (60초 운전/540초 휴지)
    cycle_phase = (t_hours * 3600) % INTERVAL / INTERVAL
    cycle_factor = 1.0 if cycle_phase < 0.1 else 0.8

    temp_front = base_temp + operating_temp * cycle_factor + fault_heat * 0.8
    temp_rear  = base_temp + operating_temp * cycle_factor + fault_heat * 1.0

    noise = np.random.randn() * 2
    return float(temp_front + noise), float(temp_rear + noise)


def simulate_torque(health_ratio, fault_mode):
    """토크 시뮬레이션"""
    base_torque = -8.0 - health_ratio * 14.0  # 정상: -8Nm, EOL: -22Nm
    if fault_mode == 'sudden' and health_ratio > 0.8:
        # 급격 스파이크
        spike = np.random.choice([-5, -3, 0], p=[0.3, 0.3, 0.4])
        base_torque += spike
    noise = np.random.randn() * 1.5
    return float(base_torque + noise)


class BearingSimulator:
    """
    베어링 Run-to-Failure 시뮬레이터

    4개 Train 베어링 생성:
      Train1: wideband  ~21h
      Train2: surface   ~19h
      Train3: cage      ~15h
      Train4: sudden    ~23h
    """
    BEARING_CONFIGS = {
        'Train1': {'mode': 'wideband', 'lifetime_h': 21.0, 'rpm_base': 800},
        'Train2': {'mode': 'surface',  'lifetime_h': 19.0, 'rpm_base': 850},
        'Train3': {'mode': 'cage',     'lifetime_h': 15.0, 'rpm_base': 900},
        'Train4': {'mode': 'sudden',   'lifetime_h': 23.0, 'rpm_base': 750},
    }

    def __init__(self, out_dir: str, n_channels: int = 4,
                 fpt_ratio: float = 0.3, seed: int = 42):
        """
        out_dir     : 데이터 저장 경로
        n_channels  : 가속도계 채널 수
        fpt_ratio   : 전체 수명 중 정상 구간 비율 (FPT 이전)
        seed        : 재현성용 랜덤 시드
        """
        np.random.seed(seed)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_channels = n_channels
        self.fpt_ratio = fpt_ratio

    def generate_all(self):
        """4개 Train 베어링 + 2개 Validation 베어링 생성"""
        summaries = []
        for name, cfg in self.BEARING_CONFIGS.items():
            print(f"  ▶ {name} 생성 중 ({cfg['mode']}, {cfg['lifetime_h']}h)...")
            summary = self._generate_bearing(name, cfg)
            summaries.append(summary)
            print(f"    → {summary['n_measurements']} 측정, EOL={summary['eol_seconds']:.0f}s")

        # Validation 베어링 (수명 모름, 중간에서 자름)
        val_configs = {
            'Val1': {'mode': 'wideband', 'lifetime_h': 18.0, 'rpm_base': 820, 'cut_ratio': 0.6},
            'Val2': {'mode': 'cage',     'lifetime_h': 20.0, 'rpm_base': 880, 'cut_ratio': 0.5},
        }
        for name, cfg in val_configs.items():
            print(f"  ▶ {name} 생성 중 (validation)...")
            cut = cfg.pop('cut_ratio')
            summary = self._generate_bearing(name, cfg, cut_ratio=cut)
            summaries.append(summary)

        # 요약 저장
        df = pd.DataFrame(summaries)
        df.to_csv(self.out_dir / 'dataset_summary.csv', index=False)
        print(f"\n✅ 데이터셋 생성 완료: {self.out_dir}")
        return df

    def _generate_bearing(self, name: str, cfg: dict, cut_ratio: float = 1.0):
        """단일 베어링 시뮬레이션"""
        lifetime_s = cfg['lifetime_h'] * 3600
        eol_s = lifetime_s * cut_ratio
        fault_mode = cfg['mode']
        base_rpm = cfg['rpm_base']
        fpt_s = lifetime_s * self.fpt_ratio   # FPT (First Prediction Time)

        bearing_dir = self.out_dir / name
        bearing_dir.mkdir(exist_ok=True)

        vibration_records = []
        operating_records = []
        t = 0.0
        meas_idx = 0

        while t < eol_s:
            t_hours = t / 3600
            rpm = rpm_schedule(t_hours, base_rpm)

            # 건강 비율 (FPT 이전: 선형 완만, FPT 이후: 가속 열화)
            if t < fpt_s:
                health_ratio = (t / fpt_s) * 0.15   # 정상 구간: 최대 0.15
            else:
                prog = (t - fpt_s) / (lifetime_s - fpt_s)
                health_ratio = 0.15 + 0.85 * (prog ** 1.5)

            health_ratio = min(health_ratio, 1.0)

            # 진동 신호 생성 (4채널)
            signals = np.stack([
                generate_bearing_signal(t, rpm, health_ratio, fault_mode)
                for _ in range(self.n_channels)
            ])  # [4, n_samples]

            # 운영 신호
            temp_f, temp_r = simulate_temperature(t_hours, health_ratio, fault_mode)
            torque = simulate_torque(health_ratio, fault_mode)

            # RUL 레이블
            rul_seconds = max(0.0, lifetime_s - t) if cut_ratio == 1.0 else None

            vibration_records.append({
                'meas_idx': meas_idx,
                't_seconds': t,
                'rpm': rpm,
                'health_ratio': health_ratio,
                'signal_ch1': signals[0],
                'signal_ch2': signals[1],
                'signal_ch3': signals[2],
                'signal_ch4': signals[3],
            })

            operating_records.append({
                'meas_idx': meas_idx,
                't_seconds': t,
                'rpm': rpm,
                'torque': torque,
                'temp_front': temp_f,
                'temp_rear': temp_r,
                'health_ratio': health_ratio,
                'rul_seconds': rul_seconds,
                'is_healthy': t < fpt_s,
            })

            # EOL 도달 확인
            if torque <= EOL_TORQUE or temp_f >= 200 or temp_r >= 200:
                break

            meas_idx += 1
            t += INTERVAL   # 다음 측정 (10분 후)

        # 진동 신호 저장 (numpy binary)
        n_meas = len(vibration_records)
        n_samples = FS * MEAS_DUR
        signals_arr = np.zeros((n_meas, 4, n_samples), dtype=np.float32)
        for i, rec in enumerate(vibration_records):
            signals_arr[i, 0] = rec['signal_ch1']
            signals_arr[i, 1] = rec['signal_ch2']
            signals_arr[i, 2] = rec['signal_ch3']
            signals_arr[i, 3] = rec['signal_ch4']

        np.save(bearing_dir / 'vibration.npy', signals_arr)

        # 운영 신호 CSV
        op_df = pd.DataFrame(operating_records).drop(
            columns=['signal_ch1', 'signal_ch2', 'signal_ch3', 'signal_ch4'],
            errors='ignore'
        )
        op_cols = ['meas_idx', 't_seconds', 'rpm', 'torque',
                   'temp_front', 'temp_rear', 'health_ratio', 'rul_seconds', 'is_healthy']
        op_df[op_cols].to_csv(bearing_dir / 'operating.csv', index=False)

        return {
            'name': name,
            'fault_mode': fault_mode,
            'lifetime_h': cfg['lifetime_h'],
            'eol_seconds': min(t, eol_s),
            'n_measurements': n_meas,
            'fpt_index': int(n_meas * self.fpt_ratio),
        }


if __name__ == '__main__':
    print("=" * 60)
    print("베어링 합성 데이터 생성기")
    print("NSK 30306 | 25.6kHz | 4ch | 60s/600s 측정 주기")
    print("=" * 60)

    sim = BearingSimulator(
        out_dir=str(Path(__file__).parent / 'raw'),
        n_channels=4,
        fpt_ratio=0.3,
        seed=42
    )
    df = sim.generate_all()
    print("\n데이터셋 요약:")
    print(df.to_string(index=False))
