"""21_VMD_CBAM_FeatureDenoising.

VMD-lite + CBAM-style attention diagnostic.
True VMD is expensive and not guaranteed to be installed, so this experiment
uses fixed band-pass modes as a VMD-lite decomposition, then applies channel and
band attention from envelope kurtosis/RMS. It creates denoised degradation
features and a nearest-neighbor RUL candidate.
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
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import FS, TRAIN_NAMES, VAL_NAMES, load_bearing

RESULT_DIR = result_dir("21_VMD_CBAM_FeatureDenoising")
BANDS = [(500, 2000), (1000, 4000), (2000, 6000), (4000, 9000)]


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def band_features(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kurts, rms = [], []
    nyq = FS / 2
    for lo, hi in BANDS:
        sos = butter(4, [lo / nyq, min(hi, nyq * 0.98) / nyq], btype="band", output="sos")
        filt = sosfiltfilt(sos, sig.astype(np.float64, copy=False))
        env = np.abs(hilbert(filt))
        kurts.append(float(np.nan_to_num(sp_kurt(env), nan=0.0)))
        rms.append(float(np.sqrt(np.mean(env ** 2))))
    return np.array(kurts), np.array(rms)


def extract_one(s4: np.ndarray) -> dict[str, float]:
    ch_scores = []
    out = {}
    for ch in range(4):
        k, r = band_features(s4[ch])
        bw = softmax(np.maximum(k, 0.0) + np.log1p(r))
        score = float(np.sum(bw * np.log1p(np.maximum(k, 0.0)) * np.log1p(r)))
        ch_scores.append(score)
        out[f"vmdlite_ch{ch}_score"] = score
        out[f"vmdlite_ch{ch}_max_kurt"] = float(k.max())
        out[f"vmdlite_ch{ch}_att_band"] = float(np.argmax(bw))
    ch_scores = np.array(ch_scores)
    cw = softmax(ch_scores)
    out["vmd_cbam_score"] = float(np.sum(cw * ch_scores))
    out["vmd_cbam_chmax"] = float(ch_scores.max())
    out["vmd_cbam_chstd"] = float(ch_scores.std())
    out["vmd_cbam_top_channel"] = float(np.argmax(cw))
    return out


def build_features() -> pd.DataFrame:
    rows = []
    for name in TRAIN_NAMES + VAL_NAMES:
        sigs, op = load_bearing(name)
        for i in range(len(op)):
            row = {"bearing": name, "measurement": int(op.iloc[i].measurement), "t_s": float(op.iloc[i].t_seconds)}
            if "rul_seconds" in op.columns and pd.notna(op.iloc[i].rul_seconds):
                row["rul_s"] = float(op.iloc[i].rul_seconds)
            row.update(extract_one(sigs[i]))
            rows.append(row)
        print(f"  extracted {name}: {len(op)} rows", flush=True)
    df = pd.DataFrame(rows).fillna(0)
    for c in ["vmd_cbam_score", "vmd_cbam_chmax", "vmd_cbam_chstd"]:
        for name in TRAIN_NAMES + VAL_NAMES:
            m = df.bearing == name
            v = df.loc[m, c].values
            df.loc[m, c + "_rel"] = (v - v[:10].mean()) / (v[:10].std() + 1e-6)
    return df


def make_candidate(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("vmd_")]
    train = df[df.bearing.isin(TRAIN_NAMES)].copy()
    test_last = df[df.bearing.isin(VAL_NAMES)].groupby("bearing", sort=False).tail(1).copy()
    sc = StandardScaler().fit(train[cols].values)
    xt = sc.transform(train[cols].values)
    xq = sc.transform(test_last[cols].values)
    dist = pairwise_distances(xq, xt)
    rows = []
    for i, (_, q) in enumerate(test_last.iterrows()):
        idx = np.argsort(dist[i])[:12]
        nn = train.iloc[idx]
        w = 1 / (dist[i, idx] + 1e-6)
        rul = nn.rul_s.values.astype(float)
        pred = float(np.quantile(rul, 0.35))
        rows.append({
            "Bearing": q.bearing,
            "RUL_pred_seconds": max(600.0, pred),
            "RUL_pred_hours": max(600.0, pred) / 3600,
            "NN1_bearing": nn.iloc[0].bearing,
            "NN1_rul_s": float(nn.iloc[0].rul_s),
            "p_eol_6000": float(np.sum(w * (rul <= 6000)) / w.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("21_VMD_CBAM_FeatureDenoising")
    df = build_features()
    df.to_csv(RESULT_DIR / "21_vmd_cbam_features.csv", index=False)
    sub = make_candidate(df)
    sub.to_csv(RESULT_DIR / "21_vmd_cbam_candidate.csv", index=False)
    sub.to_excel(RESULT_DIR / "21_vmd_cbam_submission.xlsx", index=False)
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
