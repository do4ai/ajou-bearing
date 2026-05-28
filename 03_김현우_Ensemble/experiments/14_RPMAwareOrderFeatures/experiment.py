"""14_RPMAwareOrderFeatures.

Recompute bearing fault order features on the correct physical order axis:
  order = envelope_frequency_hz / (rpm / 60).

The earlier pipelines kept legacy behavior for reproducibility. This file adds a
standalone corrected feature table for downstream similarity/calibration without
touching existing model code.

Outputs:
  results/14_rpm_order_features.csv
  results/14_rpm_order_summary.csv
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis as sp_kurt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import FS, ORDERS, TRAIN_NAMES, VAL_NAMES, load_bearing  # noqa: E402

RESULT_DIR = result_dir("14_RPMAwareOrderFeatures")
OUT_CSV = RESULT_DIR / "14_rpm_order_features.csv"
SUMMARY_CSV = RESULT_DIR / "14_rpm_order_summary.csv"

FAULTS = ["BPFI", "BPFO", "BSF", "FTF"]
HARMONICS = [1, 2, 3]
ORDER_BW = 0.15


def safe_float(v, default: float) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def envelope_order_features(sig: np.ndarray, rpm: float) -> dict[str, float]:
    sig = sig.astype(np.float64, copy=False)
    nyq = FS / 2.0
    sos = butter(4, [1000 / nyq, 6000 / nyq], btype="band", output="sos")
    filt = sosfiltfilt(sos, sig)
    env = np.abs(hilbert(filt))
    env = env - np.mean(env)
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), d=1.0 / FS)
    shaft_hz = max(rpm / 60.0, 1e-6)
    orders = freqs / shaft_hz

    valid = (orders >= 0.2) & (orders <= 35.0)
    floor_mask = valid.copy()
    for fault in FAULTS:
        base = ORDERS[fault]
        for h in HARMONICS:
            floor_mask &= ~((orders >= base * h - ORDER_BW) & (orders <= base * h + ORDER_BW))
    floor = float(np.median(spec[floor_mask])) + 1e-18 if floor_mask.any() else float(np.median(spec[valid])) + 1e-18

    out = {
        "order_env_rms": float(np.sqrt(np.mean(env ** 2))),
        "order_env_kurt": float(sp_kurt(env)),
        "order_floor": floor,
    }
    for fault in FAULTS:
        base = ORDERS[fault]
        total = 0.0
        hvals = []
        for h in HARMONICS:
            mask = (orders >= base * h - ORDER_BW) & (orders <= base * h + ORDER_BW)
            e = float(np.sum(spec[mask])) if mask.any() else 0.0
            hvals.append(e)
            total += e
            out[f"order_{fault.lower()}_h{h}_e"] = e
            out[f"order_{fault.lower()}_h{h}_snr"] = e / floor
        out[f"order_{fault.lower()}_e"] = total
        out[f"order_{fault.lower()}_snr"] = total / floor
        out[f"order_{fault.lower()}_h_ratio"] = hvals[0] / (hvals[1] + hvals[2] + 1e-18)
    return out


def extract_bearing(name: str) -> pd.DataFrame:
    sigs, op = load_bearing(name)
    rows = []
    for i in range(len(op)):
        r = op.iloc[i]
        rpm = safe_float(r.get("rpm", np.nan), 800.0)
        per_ch = []
        row = {
            "bearing": name,
            "measurement": int(r.get("measurement", i)),
            "t_s": safe_float(r.get("t_seconds", i * 600), i * 600.0),
            "rpm_used": rpm,
            "torque": safe_float(r.get("torque", np.nan), -5.0),
            "temp_front": safe_float(r.get("temp_front", np.nan), 30.0),
            "temp_rear": safe_float(r.get("temp_rear", np.nan), 30.0),
        }
        if "rul_seconds" in op.columns and pd.notna(r.get("rul_seconds", np.nan)):
            row["rul_s"] = float(r["rul_seconds"])
        for ch in range(sigs.shape[1]):
            f = envelope_order_features(sigs[i, ch], rpm)
            per_ch.append(f)
            for k, v in f.items():
                row[f"ch{ch}_{k}"] = v
        keys = list(per_ch[0].keys())
        for k in keys:
            vals = np.array([p[k] for p in per_ch], dtype=np.float64)
            row[f"order_chsym_max_{k}"] = float(vals.max())
            row[f"order_chsym_min_{k}"] = float(vals.min())
            row[f"order_chsym_range_{k}"] = float(vals.max() - vals.min())
            row[f"order_chsym_std_{k}"] = float(vals.std())
            row[f"order_chsym_top2_{k}"] = float(np.sort(vals)[-2:].mean())
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(op):
            print(f"  {name}: {i + 1}/{len(op)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    names = TRAIN_NAMES + VAL_NAMES
    frames = []
    print("14_RPMAwareOrderFeatures")
    for name in names:
        frames.append(extract_bearing(name))
    out = pd.concat(frames, ignore_index=True).fillna(0)
    out.to_csv(OUT_CSV, index=False)

    summary_cols = [
        "bearing", "measurement", "t_s", "rul_s", "rpm_used",
        "order_chsym_max_order_bpfo_snr", "order_chsym_max_order_bpfi_snr",
        "order_chsym_max_order_bsf_snr", "order_chsym_max_order_ftf_snr",
        "order_chsym_max_order_env_kurt",
    ]
    existing = [c for c in summary_cols if c in out.columns]
    summary = out.groupby("bearing", sort=False).tail(1)[existing].copy()
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"  rows={len(out)} cols={len(out.columns)}")
    print("  Last-state summary:")
    print(summary.to_string(index=False))
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
