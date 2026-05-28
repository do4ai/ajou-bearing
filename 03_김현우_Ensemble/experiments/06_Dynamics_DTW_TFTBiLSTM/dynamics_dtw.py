"""v25: dynamics features + DTW trajectory sanity check.

This is a fast diagnostic experiment before full retraining.
Goal: validate whether Test5's high HI genuinely resembles Train EOL trajectories.

Outputs:
  results/v25_dynamics_dtw.csv
  results/v25_test5_sanity.csv
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, warnings, gc
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from shared.utils import load_bearing, TRAIN_NAMES, VAL_NAMES, FS, ORDERS, asym_score

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib


METHOD_DIR = Path(__file__).resolve().parent
MODEL_DIR = METHOD_DIR / "models_v22"
RESULT_DIR = METHOD_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
WINDOW = 50


def bp(s, lo=1000, hi=6000, fs=FS):
    nyq = fs / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, s)


def fast_kurt(sig, fs=FS):
    sp = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
    nyq = fs / 2
    bands = [(500, 2000), (1000, 4000), (2000, 6000),
             (3000, 8000), (4000, 10000), (5000, 12000)]
    bk, bfc, bbw = -np.inf, 2000.0, 2000.0
    for lo, hi in bands:
        if hi >= nyq:
            continue
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 5:
            continue
        band_sp = np.sqrt(sp[mask])
        m2 = np.mean(band_sp ** 2)
        m4 = np.mean(band_sp ** 4)
        kr = m4 / (m2 ** 2 + 1e-12) - 3
        if kr > bk:
            bk, bfc, bbw = kr, (lo + hi) / 2, hi - lo
    return bfc, bbw, bk


def ch_feats(s, prefix):
    s = s.astype(np.float64)
    rms = float(np.sqrt(np.mean(s ** 2)))
    std = float(np.std(s))
    k = float(sp_kurt(s))
    sk = float(sp_skew(s))
    pk = float(np.max(np.abs(s)))
    crest = pk / (rms + 1e-10)
    p2p = float(np.ptp(s))
    shape_f = rms / (np.mean(np.abs(s)) + 1e-10)
    try:
        fc, bw, sk_kurt = fast_kurt(s)
        nyq = FS / 2
        lo = max(fc - bw / 2, 10) / nyq
        hi = min(fc + bw / 2, nyq * 0.99) / nyq
        if 0 < lo < hi < 1:
            b, a = butter(4, [lo, hi], btype="band")
            filt = filtfilt(b, a, s)
        else:
            filt = bp(s)
            sk_kurt = 0.0
    except Exception:
        filt = bp(s)
        fc, bw, sk_kurt = 3000.0, 2000.0, 0.0
    env = np.abs(hilbert(filt))
    del filt
    env_rms = float(np.sqrt(np.mean(env ** 2)))
    env_kurt = float(sp_kurt(env))
    sp2 = np.abs(np.fft.rfft(env)) / len(env)
    ords = np.fft.rfftfreq(len(env), d=1 / 256)
    nf = float(np.mean(sp2[ords > 12])) + 1e-12
    out = {
        f"{prefix}_rms": rms, f"{prefix}_std": std, f"{prefix}_kurt": k,
        f"{prefix}_skew": sk, f"{prefix}_peak": pk, f"{prefix}_crest": crest,
        f"{prefix}_p2p": p2p, f"{prefix}_shape_f": shape_f,
        f"{prefix}_fc": fc, f"{prefix}_bw": bw, f"{prefix}_sk_kurt": sk_kurt,
        f"{prefix}_env_rms": env_rms, f"{prefix}_env_kurt": env_kurt,
    }
    for nm, o in ORDERS.items():
        e1 = float(np.sum(sp2[(ords >= o - 0.15) & (ords <= o + 0.15)] ** 2))
        e2 = float(np.sum(sp2[(ords >= 2 * o - 0.15) & (ords <= 2 * o + 0.15)] ** 2))
        e3 = float(np.sum(sp2[(ords >= 3 * o - 0.15) & (ords <= 3 * o + 0.15)] ** 2))
        out[f"{prefix}_{nm.lower()}_e"] = e1 + e2 + e3
        out[f"{prefix}_{nm.lower()}_snr"] = (e1 + e2 + e3) / nf
        out[f"{prefix}_{nm.lower()}_h_ratio"] = e1 / (e2 + e3 + 1e-12)
    return out


def extract_v22(s4, rpm, torque, tf, tr):
    feats, per_ch = {}, {}
    for ci in range(4):
        f = ch_feats(s4[ci], f"ch{ci}")
        feats.update(f)
        per_ch[ci] = f
    common_keys = ["rms", "std", "kurt", "skew", "peak", "crest", "p2p",
                   "env_rms", "env_kurt", "sk_kurt"]
    for nm in ORDERS:
        common_keys.extend([f"{nm.lower()}_e", f"{nm.lower()}_snr", f"{nm.lower()}_h_ratio"])
    for k in common_keys:
        vals = np.array([per_ch[ci][f"ch{ci}_{k}"] for ci in range(4)], dtype=np.float64)
        feats[f"chsym_max_{k}"] = float(vals.max())
        feats[f"chsym_min_{k}"] = float(vals.min())
        feats[f"chsym_range_{k}"] = float(vals.max() - vals.min())
        feats[f"chsym_std_{k}"] = float(vals.std())
        feats[f"chsym_top2_{k}"] = float(np.sort(vals)[-2:].mean())
    s_all = s4.astype(np.float64)
    feats["rms_multi"] = float(np.sqrt(np.mean(s_all ** 2)))
    feats["std_multi"] = float(np.std(s_all))
    feats["peak_multi"] = float(np.max(np.abs(s_all)))
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        c = np.corrcoef(s4[i].astype(np.float64), s4[j].astype(np.float64))[0, 1]
        feats[f"corr_{i}{j}"] = float(c) if np.isfinite(c) else 0.0
    energies = np.array([float(np.mean(s4[i].astype(np.float64) ** 2)) for i in range(4)])
    feats["energy_max"] = float(energies.max())
    feats["energy_min"] = float(energies.min())
    feats["energy_ratio"] = float(energies.max() / (energies.min() + 1e-10))
    feats["energy_std"] = float(energies.std())
    rpm_v = float(rpm) if rpm == rpm else 800.0
    tq_v = float(torque) if torque == torque else -5.0
    tf_v = float(tf) if tf == tf else 30.0
    tr_v = float(tr) if tr == tr else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v), "temp_diff": tf_v - tr_v,
                  "temp_max": max(tf_v, tr_v), "temp_ratio": tf_v / (tr_v + 1e-6)})
    return feats


class DTCVAE(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU())
        self.mu = nn.Linear(32, latent)
        self.lv = nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.GELU(),
                                 nn.Linear(32, 64), nn.GELU(),
                                 nn.Linear(64, 128), nn.GELU(), nn.Linear(128, d))

    def fwd(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
        return self.dec(z), mu, lv, z


def rolling_slope(values, win):
    out = np.zeros(len(values), dtype=np.float64)
    xs = np.arange(win, dtype=np.float64)
    xs = (xs - xs.mean()) / (xs.std() + 1e-12)
    for i in range(len(values)):
        st = max(0, i - win + 1)
        seg = values[st:i + 1]
        if len(seg) < 3:
            out[i] = 0.0
            continue
        x = np.arange(len(seg), dtype=np.float64)
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (seg - np.mean(seg)) / (np.std(seg) + 1e-12)
        out[i] = float(np.mean(x * y))
    return out


def add_dynamics(df):
    dyn_cols = [c for c in ["HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt"] if c in df.columns]
    out = []
    for _, sub in df.groupby("bearing", sort=False):
        sub = sub.copy()
        for c in dyn_cols:
            v = sub[c].astype(float).values
            for lag in [1, 3, 5]:
                sub[f"{c}_d{lag}"] = pd.Series(v).diff(lag).fillna(0).values
            sub[f"{c}_slope5"] = rolling_slope(v, 5)
            sub[f"{c}_slope10"] = rolling_slope(v, 10)
            sub[f"{c}_acc"] = pd.Series(sub[f"{c}_d1"]).diff(1).fillna(0).values
            sub[f"{c}_roll_std5"] = pd.Series(v).rolling(5, min_periods=2).std().fillna(0).values
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def load_v22_features(names):
    meta = joblib.load(MODEL_DIR / "feature_meta.pkl")
    FC = meta["FC"]
    sc = joblib.load(MODEL_DIR / "scaler_features.pkl")
    vae = DTCVAE(len(FC), latent=4)
    vae.load_state_dict(torch.load(MODEL_DIR / "dtcvae.pt", map_location="cpu"))
    vae.eval()
    all_rows = []
    for nm in names:
        print(f"[extract] {nm}", flush=True)
        sigs, op = load_bearing(nm)
        rows = []
        for i in range(len(op)):
            r = op.iloc[i]
            f = extract_v22(sigs[i], r.rpm, r.torque, r.temp_front, r.temp_rear)
            f["bearing"] = nm
            f["measurement"] = int(r.measurement)
            f["t_s"] = float(r.t_seconds)
            f["rul_s"] = float(r.rul_seconds) if r.rul_seconds == r.rul_seconds else np.nan
            rows.append(f)
        df = pd.DataFrame(rows)
        for c in FC:
            if c not in df.columns:
                df[c] = 0.0
        X = sc.transform(df[FC].fillna(0).values)
        with torch.no_grad():
            _, _, _, z = vae.fwd(torch.tensor(X, dtype=torch.float32))
        hi = z[:, 0].numpy()
        hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
        df["HI"] = hi
        for k in range(z.shape[1]):
            df[f"latent_{k}"] = z[:, k].numpy()
        all_rows.append(df)
        del sigs
        gc.collect()
    return pd.concat(all_rows, ignore_index=True)


def dtw_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def zfit_transform(train_df, test_df, cols):
    sc = StandardScaler().fit(train_df[cols].fillna(0).values)
    return sc.transform(train_df[cols].fillna(0).values), sc.transform(test_df[cols].fillna(0).values)


def run_dtw(df):
    traj_cols = [c for c in ["HI", "rms_multi", "energy_ratio", "chsym_max_env_kurt",
                             "HI_slope5", "HI_d5"] if c in df.columns]
    train_df = df[df.bearing.isin(TRAIN_NAMES)].copy()
    records = []
    for test_nm in VAL_NAMES:
        test_df = df[df.bearing == test_nm].copy()
        # scale using train only, then split arrays back by bearing
        _, Xt = zfit_transform(train_df, test_df, traj_cols)
        test_seq = Xt[-min(WINDOW, len(Xt)):]
        best = None
        for tr_nm in TRAIN_NAMES:
            sub = train_df[train_df.bearing == tr_nm].copy()
            Xtr, _ = zfit_transform(sub, test_df, traj_cols)
            if len(Xtr) < len(test_seq):
                continue
            for end in range(len(test_seq), len(Xtr) + 1):
                win = Xtr[end - len(test_seq):end]
                d = dtw_distance(test_seq, win)
                r = sub.iloc[end - 1]
                rec = {
                    "Bearing": test_nm,
                    "match_train": tr_nm,
                    "match_end_idx": int(end - 1),
                    "match_t_s": float(r.t_s),
                    "match_rul_s": float(r.rul_s),
                    "dtw_distance": d,
                    "test_HI_last": float(test_df.HI.iloc[-1]),
                    "test_HI_slope5_last": float(test_df.HI_slope5.iloc[-1]) if "HI_slope5" in test_df else np.nan,
                    "test_HI_d5_last": float(test_df.HI_d5.iloc[-1]) if "HI_d5" in test_df else np.nan,
                }
                if best is None or d < best["dtw_distance"]:
                    best = rec
        if best:
            records.append(best)
    return pd.DataFrame(records).sort_values("Bearing")


def main():
    print("=" * 70)
    print("v25 dynamics + DTW sanity check")
    print("=" * 70)
    df = load_v22_features(TRAIN_NAMES + VAL_NAMES)
    df = add_dynamics(df)
    dyn_out = RESULT_DIR / "v25_features_dynamics.parquet"
    try:
        df.to_parquet(dyn_out, index=False)
        print(f"saved {dyn_out}")
    except Exception:
        csv_out = RESULT_DIR / "v25_features_dynamics.csv"
        df.to_csv(csv_out, index=False)
        print(f"saved {csv_out}")

    dtw = run_dtw(df)
    # Merge current v19/v24 submissions for sanity comparison.
    v24 = pd.read_csv(RESULT_DIR / "submission_v24_v17v22_debug.csv")
    v19_path = RESULT_DIR / "submission_v19_blend_debug.csv"
    if v19_path.exists():
        v19 = pd.read_csv(v19_path)[["Bearing", "RUL_blend_combined_s"]].rename(columns={"RUL_blend_combined_s": "RUL_v19_combined_s"})
        dtw = dtw.merge(v19, on="Bearing", how="left")
    v24 = v24[["Bearing", "HI_last", "RUL_v17_s", "RUL_v18_s", "RUL_blend_combined_s"]].rename(
        columns={"RUL_v18_s": "RUL_v22_s", "RUL_blend_combined_s": "RUL_v24_combined_s"})
    dtw = dtw.merge(v24, on="Bearing", how="left")
    out = RESULT_DIR / "v25_dynamics_dtw.csv"
    dtw.to_csv(out, index=False)
    print(f"\nDTW summary -> {out}")
    print(dtw.to_string(index=False))

    test5 = dtw[dtw.Bearing == "Test5"].copy()
    test5_out = RESULT_DIR / "v25_test5_sanity.csv"
    test5.to_csv(test5_out, index=False)
    if len(test5):
        r = test5.iloc[0]
        print("\nTest5 sanity:")
        print(f"  nearest train window: {r.match_train} idx={int(r.match_end_idx)} rul={r.match_rul_s:.0f}s")
        print(f"  v24={r.RUL_v24_combined_s:.0f}s  v19={r.get('RUL_v19_combined_s', np.nan):.0f}s")
        print(f"  distance={r.dtw_distance:.4f} HI={r.HI_last:.3f}")


if __name__ == "__main__":
    main()
