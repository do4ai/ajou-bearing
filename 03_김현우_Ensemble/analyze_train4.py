"""Train4 정밀 분석: 왜 모든 모델이 약한가?

분석 항목:
  1. HI 곡선 패턴 (다른 베어링과 비교)
  2. 운전 조건 분포 (rpm, torque)
  3. 진동 특징 진행 (RMS, kurtosis)
  4. v17 vs v18 예측 분산
  5. 시퀀스 변화 시점 (도약 지점)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import load_bearing, TRAIN_NAMES, FS

import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis as sp_kurt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = Path(__file__).resolve().parent / "results"

# 베어링별 RMS, kurtosis 진행 차트
print("=" * 70); print("  Train1~4 비교 분석"); print("=" * 70)
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
stats = {}
for ci, nm in enumerate(TRAIN_NAMES):
    sigs, op = load_bearing(nm)
    n = len(op); t_h = op.t_seconds.values / 3600
    rms_ch = np.zeros((n, 4)); kurt_ch = np.zeros((n, 4)); peak_ch = np.zeros((n, 4))
    for i in range(n):
        for c in range(4):
            s = sigs[i, c].astype(np.float64)
            rms_ch[i, c] = np.sqrt(np.mean(s**2))
            kurt_ch[i, c] = sp_kurt(s)
            peak_ch[i, c] = np.max(np.abs(s))
    stats[nm] = {"rms": rms_ch, "kurt": kurt_ch, "peak": peak_ch,
                 "rpm": op.rpm.values, "tq": op.torque.values,
                 "tf": op.temp_front.values, "tr": op.temp_rear.values,
                 "t_h": t_h, "rul": op.rul_seconds.values}
    del sigs
    # plot
    ax = axes[0, ci]
    for c in range(4):
        ax.plot(t_h, rms_ch[:, c], label=f"ch{c}", alpha=0.7, lw=1.5)
    ax.set(xlabel="Time(h)", ylabel="RMS", title=f"{nm} RMS (4ch)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1, ci]
    for c in range(4):
        ax.plot(t_h, kurt_ch[:, c], label=f"ch{c}", alpha=0.7, lw=1.5)
    ax.set(xlabel="Time(h)", ylabel="Kurtosis", title=f"{nm} Kurt (4ch)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[2, ci]
    ax.plot(t_h, op.rpm.values, color="blue", label="rpm")
    ax2 = ax.twinx()
    ax2.plot(t_h, op.torque.values, color="red", label="torque", lw=1)
    ax.set(xlabel="Time(h)", ylabel="rpm (blue)", title=f"{nm} Operating")
    ax2.set_ylabel("torque (red)")
    ax = axes[3, ci]
    ax.plot(t_h, op.temp_front.values, label="temp_front", color="orange")
    ax.plot(t_h, op.temp_rear.values, label="temp_rear", color="purple")
    ax.set(xlabel="Time(h)", ylabel="Temp", title=f"{nm} Temp")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULT_DIR / "train4_compare.png", dpi=100); plt.close()
print(f"  → {RESULT_DIR}/train4_compare.png")


# 통계 비교: 각 베어링의 EOL 근처 RMS 도약 분석
print("\n[RMS 도약 분석] 마지막 N개 vs 처음 N개 비율:")
print(f"  {'name':10s} {'len':5s} {'rul_h':6s} {'rms_start':12s} {'rms_end':12s} {'ratio':6s}  {'kurt_end':10s}")
for nm in TRAIN_NAMES:
    s = stats[nm]
    n = len(s["t_h"]); k = max(5, n//10)
    rms_start = s["rms"][:k].mean(); rms_end = s["rms"][-k:].mean()
    kurt_end = s["kurt"][-k:].mean()
    print(f"  {nm:10s} {n:5d} {s['t_h'][-1]+0.17:6.1f} {rms_start:12.4f} {rms_end:12.4f} {rms_end/rms_start:6.2f}  {kurt_end:10.2f}")


# v17 / v18 예측 비교 시각화 (이미 있음)
# 추가: Train4 만 자세히 — model variance 시각화

# 운전 조건 변화 분석
print("\n[운전 조건 통계]")
for nm in TRAIN_NAMES:
    s = stats[nm]
    print(f"  {nm}: rpm μ={s['rpm'].mean():.0f} σ={s['rpm'].std():.0f}  "
          f"torque μ={s['tq'].mean():.2f} σ={s['tq'].std():.2f}  "
          f"tf μ={s['tf'].mean():.1f} tr μ={s['tr'].mean():.1f}")
