"""v19: v17 + v18 HI-conditioned 블렌딩 + β보정 + submission 생성.

v17: mid-life 강 (full 0.59) — HI 낮을 때 사용
v18: EOL 강 (last 0.96) — HI 높을 때 사용

블렌딩: pred = v17 * w_low + v18 * w_high
  w_high = sigmoid((HI - thr) * slope)
  w_low  = 1 - w_high

그리드서치 영역:
  thr   ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
  slope ∈ {5, 10, 20, 40}
  β     ∈ {0.85, 0.90, 0.95, 1.00, 1.05} (v17 + v18 블렌드 후 적용)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, load_bearing, TRAIN_NAMES, VAL_NAMES, FS, ORDERS

import torch, torch.nn as nn
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
MODEL_DIR_V17 = METHOD_DIR / "models"
MODEL_DIR_V18 = METHOD_DIR / "models_v22"
RESULT_DIR = METHOD_DIR / "results"

# ── 피처 추출 — v17 (1채널 31피처) + v18 (4채널 116피처) 따로 ───────────
def bp(s, lo=1000, hi=6000, fs=FS):
    nyq = fs / 2; b, a = butter(4, [lo/nyq, hi/nyq], btype="band"); return filtfilt(b, a, s)

def fast_kurt(sig, fs=FS):
    sp = np.abs(np.fft.rfft(sig)) ** 2; freqs = np.fft.rfftfreq(len(sig), d=1/fs); nyq = fs / 2
    bands = [(500,2000),(1000,4000),(2000,6000),(3000,8000),(4000,10000),(5000,12000)]
    bk, bfc, bbw = -np.inf, 2000., 2000.
    for lo, hi in bands:
        if hi >= nyq: continue
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 5: continue
        band_sp = np.sqrt(sp[mask]); m2 = np.mean(band_sp**2); m4 = np.mean(band_sp**4)
        kr = m4 / (m2**2 + 1e-12) - 3
        if kr > bk: bk = kr; bfc = (lo+hi)/2; bbw = (hi-lo)
    return bfc, bbw, bk

def extract_v17(s4, rpm, torque, tf, tr):
    """v17 호환 (1채널, 31 피처)"""
    s = s4[0].astype(np.float64); s_all = s4.astype(np.float64)
    rms = float(np.sqrt(np.mean(s**2))); std = float(np.std(s))
    k = float(sp_kurt(s)); sk = float(sp_skew(s)); pk = float(np.max(np.abs(s))); crest = pk/(rms+1e-10)
    p2p = float(np.ptp(s)); shape_f = rms / (np.mean(np.abs(s)) + 1e-10)
    rms_multi = float(np.sqrt(np.mean(s_all**2)))
    try:
        fc, bw, sk_kurt = fast_kurt(s); nyq = FS/2
        lo, hi = max(fc-bw/2, 10)/nyq, min(fc+bw/2, nyq*0.99)/nyq
        if 0 < lo < hi < 1: b, a = butter(4, [lo, hi], btype="band"); filt = filtfilt(b, a, s)
        else: filt = bp(s); sk_kurt = 0.0
    except: filt = bp(s); fc = 3000.; bw = 2000.; sk_kurt = 0.0
    env = np.abs(hilbert(filt)); del filt
    env_rms = float(np.sqrt(np.mean(env**2))); env_kurt = float(sp_kurt(env))
    sp2 = np.abs(np.fft.rfft(env)) / len(env); ords = np.fft.rfftfreq(len(env), d=1/256)
    nf = float(np.mean(sp2[ords > 12])) + 1e-12
    feats = {"rms": rms, "std": std, "kurtosis": k, "skewness": sk, "peak": pk,
             "crest": crest, "p2p": p2p, "shape_f": shape_f, "rms_multi": rms_multi,
             "fc": fc, "bw": bw, "sk_kurt": sk_kurt, "env_rms": env_rms, "env_kurt": env_kurt}
    for nm, o in ORDERS.items():
        e1 = float(np.sum(sp2[(ords >= o-0.15) & (ords <= o+0.15)]**2))
        e2 = float(np.sum(sp2[(ords >= 2*o-0.15) & (ords <= 2*o+0.15)]**2))
        e3 = float(np.sum(sp2[(ords >= 3*o-0.15) & (ords <= 3*o+0.15)]**2))
        feats[f"{nm.lower()}_e"] = e1 + e2 + e3
        feats[f"{nm.lower()}_snr"] = (e1 + e2 + e3) / nf
        feats[f"{nm.lower()}_h_ratio"] = e1 / (e2 + e3 + 1e-12)
    rpm_v = float(rpm) if (rpm == rpm) else 800.0
    tq_v = float(torque) if (torque == torque) else -5.0
    tf_v = float(tf) if (tf == tf) else 30.0
    tr_v = float(tr) if (tr == tr) else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v)})
    return feats


def ch_feats(s, prefix):
    s = s.astype(np.float64)
    rms = float(np.sqrt(np.mean(s**2))); std = float(np.std(s))
    k = float(sp_kurt(s)); sk = float(sp_skew(s))
    pk = float(np.max(np.abs(s))); crest = pk / (rms + 1e-10)
    p2p = float(np.ptp(s)); shape_f = rms / (np.mean(np.abs(s)) + 1e-10)
    try:
        fc, bw, sk_kurt = fast_kurt(s); nyq = FS/2
        lo, hi = max(fc-bw/2, 10)/nyq, min(fc+bw/2, nyq*0.99)/nyq
        if 0 < lo < hi < 1: b, a = butter(4, [lo, hi], btype="band"); filt = filtfilt(b, a, s)
        else: filt = bp(s); sk_kurt = 0.0
    except: filt = bp(s); fc = 3000.; bw = 2000.; sk_kurt = 0.0
    env = np.abs(hilbert(filt)); del filt
    env_rms = float(np.sqrt(np.mean(env**2))); env_kurt = float(sp_kurt(env))
    sp2 = np.abs(np.fft.rfft(env)) / len(env); ords = np.fft.rfftfreq(len(env), d=1/256)
    nf = float(np.mean(sp2[ords > 12])) + 1e-12
    out = {f"{prefix}_rms": rms, f"{prefix}_std": std, f"{prefix}_kurt": k,
           f"{prefix}_skew": sk, f"{prefix}_peak": pk, f"{prefix}_crest": crest,
           f"{prefix}_p2p": p2p, f"{prefix}_shape_f": shape_f,
           f"{prefix}_fc": fc, f"{prefix}_bw": bw, f"{prefix}_sk_kurt": sk_kurt,
           f"{prefix}_env_rms": env_rms, f"{prefix}_env_kurt": env_kurt}
    for nm, o in ORDERS.items():
        e1 = float(np.sum(sp2[(ords >= o-0.15) & (ords <= o+0.15)]**2))
        e2 = float(np.sum(sp2[(ords >= 2*o-0.15) & (ords <= 2*o+0.15)]**2))
        e3 = float(np.sum(sp2[(ords >= 3*o-0.15) & (ords <= 3*o+0.15)]**2))
        out[f"{prefix}_{nm.lower()}_e"] = e1 + e2 + e3
        out[f"{prefix}_{nm.lower()}_snr"] = (e1 + e2 + e3) / nf
        out[f"{prefix}_{nm.lower()}_h_ratio"] = e1 / (e2 + e3 + 1e-12)
    return out

def extract_v18(s4, rpm, torque, tf, tr):
    """실제 v22 extract (채널 대칭 피처 포함)."""
    feats = {}
    per_ch = {}
    for ci in range(4):
        f = ch_feats(s4[ci], f"ch{ci}")
        feats.update(f); per_ch[ci] = f
    common_keys = ["rms","std","kurt","skew","peak","crest","p2p","env_rms","env_kurt","sk_kurt"]
    for nm, o in ORDERS.items():
        common_keys.extend([f"{nm.lower()}_e", f"{nm.lower()}_snr", f"{nm.lower()}_h_ratio"])
    for k in common_keys:
        vals = np.array([per_ch[ci][f"ch{ci}_{k}"] for ci in range(4)], dtype=np.float64)
        feats[f"chsym_max_{k}"] = float(vals.max())
        feats[f"chsym_min_{k}"] = float(vals.min())
        feats[f"chsym_range_{k}"] = float(vals.max() - vals.min())
        feats[f"chsym_std_{k}"] = float(vals.std())
        feats[f"chsym_top2_{k}"] = float(np.sort(vals)[-2:].mean())
    s_all = s4.astype(np.float64)
    feats["rms_multi"] = float(np.sqrt(np.mean(s_all**2)))
    feats["std_multi"] = float(np.std(s_all))
    feats["peak_multi"] = float(np.max(np.abs(s_all)))
    for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        c = np.corrcoef(s4[i].astype(np.float64), s4[j].astype(np.float64))[0,1]
        feats[f"corr_{i}{j}"] = float(c) if np.isfinite(c) else 0.0
    energies = np.array([float(np.mean(s4[i].astype(np.float64)**2)) for i in range(4)])
    feats["energy_max"] = float(energies.max())
    feats["energy_min"] = float(energies.min())
    feats["energy_ratio"] = float(energies.max() / (energies.min() + 1e-10))
    feats["energy_std"] = float(energies.std())
    rpm_v = float(rpm) if (rpm == rpm) else 800.0
    tq_v = float(torque) if (torque == torque) else -5.0
    tf_v = float(tf) if (tf == tf) else 30.0
    tr_v = float(tr) if (tr == tr) else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v),
                  "temp_diff": tf_v - tr_v, "temp_max": max(tf_v, tr_v),
                  "temp_ratio": tf_v / (tr_v + 1e-6)})
    return feats


# ── 모델 정의 ──────────────────────────────────────────────────────
class DTCVAE_v17(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU())
        self.mu = nn.Linear(32, latent); self.lv = nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.GELU(),
                                 nn.Linear(32, 64), nn.GELU(), nn.Linear(64, d))
    def fwd(self, x):
        h = self.enc(x); return self.dec(self.mu(h)), self.mu(h), self.lv(h), self.mu(h)

class DTCVAE_v18(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU())
        self.mu = nn.Linear(32, latent); self.lv = nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.GELU(),
                                 nn.Linear(32, 64), nn.GELU(),
                                 nn.Linear(64, 128), nn.GELU(), nn.Linear(128, d))
    def fwd(self, x):
        h = self.enc(x); return self.dec(self.mu(h)), self.mu(h), self.lv(h), self.mu(h)


class TCNBlock(nn.Module):
    def __init__(self, ic, oc, d=1):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, 5, padding=4*d, dilation=d)
        self.bn = nn.BatchNorm1d(oc); self.act = nn.GELU(); self.drop = nn.Dropout(0.2)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        o = self.conv(x)[..., :x.shape[-1]]
        return self.drop(self.act(self.bn(o))) + self.res(x)

class TFTModel(nn.Module):
    def __init__(self, fd, dm=64, nh=4, dilations=(1,2,4,8)):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(fd, fd), nn.Softmax(dim=-1))
        self.tcn = nn.Sequential(*[TCNBlock(fd if i==0 else dm, dm, d) for i, d in enumerate(dilations)])
        enc = nn.TransformerEncoderLayer(dm, nh, dim_feedforward=128, dropout=0.2,
                                          batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(dm, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        w = self.vsn(x); xw = (x * w).transpose(1, 2)
        c = self.tcn(xw).transpose(1, 2); ctx = self.tr(c)[:, -1, :]
        return self.fc(ctx).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self, fd, dm=64):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(fd, fd), nn.Softmax(dim=-1))
        self.proj = nn.Linear(fd, dm)
        self.lstm = nn.LSTM(dm, dm // 2, num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.2)
        self.attn = nn.Sequential(nn.Linear(dm, 32), nn.Tanh(), nn.Linear(32, 1))
        self.fc = nn.Sequential(nn.Linear(dm, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        w = self.vsn(x); xw = self.proj(x * w)
        o, _ = self.lstm(xw)
        a = torch.softmax(self.attn(o).squeeze(-1), dim=1).unsqueeze(-1)
        ctx = (o * a).sum(dim=1)
        return self.fc(ctx).squeeze(-1)


def cusum(hi, k=0.5, h=5., r=0.2):
    n = len(hi); n0 = max(5, int(n*r)); mu0 = hi[:n0].mean(); s0 = hi[:n0].std() + 1e-8; S = 0.
    for i in range(n0, n):
        S = max(0., S + (hi[i] - mu0) / s0 - k)
        if S > h: return i
    return int(n * 0.8)


# ── 두 버전 모두 피처 + HI 추출 ─────────────────────────────────────
def load_all_features(bearings):
    sigs_op = {nm: load_bearing(nm) for nm in bearings}

    # v17 피처
    sc17 = joblib.load(MODEL_DIR_V17 / "scaler_features.pkl")
    meta17 = joblib.load(MODEL_DIR_V17 / "feature_meta.pkl")
    FC17 = meta17["FC"]; FC_HI17 = meta17["FC_HI"]
    vae17 = DTCVAE_v17(len(FC17), latent=4); vae17.load_state_dict(torch.load(MODEL_DIR_V17 / "dtcvae.pt"))
    vae17.eval()

    # v18 피처
    sc18 = joblib.load(MODEL_DIR_V18 / "scaler_features.pkl")
    meta18 = joblib.load(MODEL_DIR_V18 / "feature_meta.pkl")
    FC18 = meta18["FC"]; FC_HI18 = meta18["FC_HI"]
    vae18 = DTCVAE_v18(len(FC18), latent=4); vae18.load_state_dict(torch.load(MODEL_DIR_V18 / "dtcvae.pt"))
    vae18.eval()

    data = {}
    for nm in bearings:
        sigs, op = sigs_op[nm]
        rows17 = []; rows18 = []
        for i in range(len(op)):
            r = op.iloc[i]
            rows17.append(extract_v17(sigs[i], r.rpm, r.torque, r.temp_front, r.temp_rear))
            rows18.append(extract_v18(sigs[i], r.rpm, r.torque, r.temp_front, r.temp_rear))
        df17 = pd.DataFrame(rows17); df18 = pd.DataFrame(rows18)
        for c in FC17:
            if c not in df17.columns: df17[c] = 0.0
        for c in FC18:
            if c not in df18.columns: df18[c] = 0.0
        X17 = sc17.transform(df17[FC17].fillna(0).values)
        X18 = sc18.transform(df18[FC18].fillna(0).values)
        with torch.no_grad():
            _, _, _, z17 = vae17.fwd(torch.tensor(X17, dtype=torch.float32))
            _, _, _, z18 = vae18.fwd(torch.tensor(X18, dtype=torch.float32))
        hi17 = z17[:,0].numpy(); hi17 = (hi17 - hi17.min()) / (hi17.max() - hi17.min() + 1e-10)
        hi18 = z18[:,0].numpy(); hi18 = (hi18 - hi18.min()) / (hi18.max() - hi18.min() + 1e-10)
        df17["HI"] = hi17
        for i in range(z17.shape[1]): df17[f"latent_{i}"] = z17[:,i].numpy()
        df18["HI"] = hi18
        for i in range(z18.shape[1]): df18[f"latent_{i}"] = z18[:,i].numpy()
        df17["t_s"] = op["t_seconds"].values; df18["t_s"] = op["t_seconds"].values
        if "rul_seconds" in op.columns:
            df17["rul_s"] = op["rul_seconds"].values; df18["rul_s"] = op["rul_seconds"].values
        data[nm] = {"df17": df17, "df18": df18, "FC_HI17": FC_HI17, "FC_HI18": FC_HI18,
                    "t_s": op["t_seconds"].values, "op": op}
    return data, meta17, meta18


def fold_preds_from(model_dir, df, FC_HI_f, fold_name, is_v18=False):
    """단일 폴드 모델로 베어링 예측."""
    fold_dir = model_dir / f"fold_{fold_name}"
    meta = joblib.load(fold_dir / "meta.pkl")
    sc2 = meta["scaler"]; rul_max = meta["rul_max"]; SEQ_LEN = meta["SEQ_LEN"]
    TFT_SEEDS = meta["TFT_SEEDS"]; BILSTM_SEEDS = meta["BILSTM_SEEDS"]
    for c in FC_HI_f:
        if c not in df.columns: df[c] = 0.0
    X = sc2.transform(df[FC_HI_f].fillna(0).values)
    if len(X) < SEQ_LEN:
        pad = np.tile(X[0:1], (SEQ_LEN - len(X), 1))
        X_pad = np.vstack([pad, X])
        vs = np.array([X_pad[i:i+SEQ_LEN] for i in range(len(X_pad)-SEQ_LEN+1)], np.float32)
        pred_len = len(X)
    else:
        vs = np.array([X[i:i+SEQ_LEN] for i in range(len(X)-SEQ_LEN+1)], np.float32)
        pred_len = len(X)

    arch_seeds = [("tft", TFTModel, s) for s in TFT_SEEDS] + \
                 [("bilstm", BiLSTMModel, s) for s in BILSTM_SEEDS]
    preds = []
    for arch_name, cls, sd in arch_seeds:
        model = cls(len(FC_HI_f))
        ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
        if not ckpt.exists(): continue
        model.load_state_dict(torch.load(ckpt)); model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(vs)).numpy() * rul_max
        full = np.zeros(pred_len, dtype=np.float32)
        offset = pred_len - len(raw)
        full[:offset] = raw[0]; full[offset:] = raw
        preds.append(full)
    return np.array(preds)  # [n_models, n_meas]


# ── 1) LOBO OOF 블렌딩 ─────────────────────────────────────────────
print("=" * 70); print("  v17 + v18 블렌딩 (LOBO 검증)"); print("=" * 70)

print("\n[1] 두 버전 피처 추출...", flush=True)
data_train, meta17, meta18 = load_all_features(TRAIN_NAMES)

print("\n[2] LOBO 예측 (각 베어링 → 자기 fold 모델로)...", flush=True)
oof = {}
for val in TRAIN_NAMES:
    d = data_train[val]
    p17 = fold_preds_from(MODEL_DIR_V17, d["df17"].copy(), d["FC_HI17"], val, is_v18=False)
    p18 = fold_preds_from(MODEL_DIR_V18, d["df18"].copy(), d["FC_HI18"], val, is_v18=True)
    # SEQ_LEN 정렬: 두 버전 모두 SEQ_LEN=10이지만 안전하게 마지막 N개 사용
    p17_med = np.median(np.clip(p17, 0, None), axis=0)
    p18_med = np.median(np.clip(p18, 600, None), axis=0)
    # HI: v18 HI 사용 (4채널 기반으로 더 신뢰)
    hi = d["df18"]["HI"].values
    yvl = d["df18"]["rul_s"].values
    # 길이 정렬
    L = min(len(p17_med), len(p18_med), len(hi), len(yvl))
    oof[val] = dict(p17=p17_med[-L:], p18=p18_med[-L:],
                    hi=hi[-L:], yvl=yvl[-L:], t_h=d["t_s"][-L:]/3600)
    print(f"  {val}: L={L}  p17_last={p17_med[-1]:.0f}  p18_last={p18_med[-1]:.0f}  true_last={yvl[-1]:.0f}",
          flush=True)


# ── 2) HI-conditioned 블렌딩 그리드서치 ────────────────────────────
def blend(p17, p18, hi, thr, slope, beta=1.0):
    w_high = 1.0 / (1.0 + np.exp(-(hi - thr) * slope))  # HI > thr → 1
    w_low = 1.0 - w_high
    p = p17 * w_low + p18 * w_high
    return np.clip(p * beta, 600, None)

print("\n[3] HI-conditioned 블렌딩 그리드서치...")
print("  (목표: 4-fold 평균 full score 최대화. last는 부수 효과)")
best_full = None; best_combo_full = None
best_combined = None; best_combo_comb = None
records = []
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
    for slope in [3, 5, 10, 20, 40]:
        for beta in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
            full_s = []; last_s = []
            for val in TRAIN_NAMES:
                o = oof[val]
                p = blend(o["p17"], o["p18"], o["hi"], thr, slope, beta)
                full_s.append(asym_score(p, o["yvl"]))
                last_s.append(asym_score(p[-1:], o["yvl"][-1:]))
            fmean = np.mean(full_s); lmean = np.mean(last_s)
            comb = 0.7 * fmean + 0.3 * lmean
            records.append((thr, slope, beta, fmean, lmean, comb))
            if best_full is None or fmean > best_full:
                best_full = fmean; best_combo_full = (thr, slope, beta, fmean, lmean)
            if best_combined is None or comb > best_combined:
                best_combined = comb; best_combo_comb = (thr, slope, beta, fmean, lmean)

rec = pd.DataFrame(records, columns=["thr","slope","beta","full","last","combined"])
rec.to_csv(RESULT_DIR / "blend_v17v22_grid.csv", index=False)
print("\n  ── Top 10 by full ──")
print(rec.nlargest(10, "full").to_string(index=False))
print("\n  ── Top 10 by combined (0.7*full + 0.3*last) ──")
print(rec.nlargest(10, "combined").to_string(index=False))


# ── 3) 베이스라인 비교 ─────────────────────────────────────────────
print("\n[4] 베이스라인 비교 ──")
v17_only_full = [asym_score(np.clip(oof[v]["p17"],600,None), oof[v]["yvl"]) for v in TRAIN_NAMES]
v17_only_last = [asym_score(np.clip(oof[v]["p17"][-1:],600,None), oof[v]["yvl"][-1:]) for v in TRAIN_NAMES]
v18_only_full = [asym_score(np.clip(oof[v]["p18"],600,None), oof[v]["yvl"]) for v in TRAIN_NAMES]
v18_only_last = [asym_score(np.clip(oof[v]["p18"][-1:],600,None), oof[v]["yvl"][-1:]) for v in TRAIN_NAMES]
print(f"  v17 only:  full={np.mean(v17_only_full):.4f}  last={np.mean(v17_only_last):.4f}")
print(f"  v18 only:  full={np.mean(v18_only_full):.4f}  last={np.mean(v18_only_last):.4f}")
print(f"  blend best full:      thr={best_combo_full[0]}  slope={best_combo_full[1]}  β={best_combo_full[2]}  "
      f"full={best_combo_full[3]:.4f}  last={best_combo_full[4]:.4f}")
print(f"  blend best combined:  thr={best_combo_comb[0]}  slope={best_combo_comb[1]}  β={best_combo_comb[2]}  "
      f"full={best_combo_comb[3]:.4f}  last={best_combo_comb[4]:.4f}")


# ── 4) 최적 조합으로 시각화 + Test 추론 ────────────────────────────
opt_thr, opt_slope, opt_beta = best_combo_full[0], best_combo_full[1], best_combo_full[2]
print(f"\n[5] 최적 (best full) thr={opt_thr} slope={opt_slope} β={opt_beta} 적용 시각화...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, val in zip(axes.flatten(), TRAIN_NAMES):
    o = oof[val]
    p = blend(o["p17"], o["p18"], o["hi"], opt_thr, opt_slope, opt_beta)
    fs = asym_score(p, o["yvl"]); ls = asym_score(p[-1:], o["yvl"][-1:])
    ax.plot(o["t_h"], o["yvl"]/3600, "k-", lw=2.5, label="TRUE")
    ax.plot(o["t_h"], o["p17"]/3600, "b-", alpha=0.5, label="v17")
    ax.plot(o["t_h"], o["p18"]/3600, "g-", alpha=0.5, label="v18")
    ax.plot(o["t_h"], p/3600, "r-", lw=2, label=f"blend(full={fs:.3f} last={ls:.3f})")
    ax.plot(o["t_h"], o["hi"]*o["yvl"].max()/3600, "orange", ls=":", alpha=0.5, label="HI scaled")
    ax.set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} pred_last={p[-1]:.0f}s true={o['yvl'][-1]:.0f}s")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle(f"v17⊕v18 HI-blend (thr={opt_thr} slope={opt_slope} β={opt_beta})  "
             f"full={best_combo_full[3]:.3f} last={best_combo_full[4]:.3f}")
plt.tight_layout()
plt.savefig(RESULT_DIR / "blend_v17v22_LOBO.png", dpi=120); plt.close()


# ── 5) Test 추론 ────────────────────────────────────────────────────
print("\n[6] Test1~6 추론 (블렌드 적용)...", flush=True)
data_test, _, _ = load_all_features(VAL_NAMES)
rows_out = []
for nm in VAL_NAMES:
    d = data_test[nm]
    # 모든 폴드 모델로 예측 (테스트는 폴드 선택 없이 모든 폴드 ensemble)
    all_p17 = []; all_p18 = []
    for fold in TRAIN_NAMES:
        p17 = fold_preds_from(MODEL_DIR_V17, d["df17"].copy(), d["FC_HI17"], fold, is_v18=False)
        p18 = fold_preds_from(MODEL_DIR_V18, d["df18"].copy(), d["FC_HI18"], fold, is_v18=True)
        all_p17.append(np.median(np.clip(p17, 0, None), axis=0))
        all_p18.append(np.median(np.clip(p18, 600, None), axis=0))
    p17_ens = np.median(np.array(all_p17), axis=0)
    p18_ens = np.median(np.array(all_p18), axis=0)
    hi = d["df18"]["HI"].values
    L = min(len(p17_ens), len(p18_ens), len(hi))
    p17_ens = p17_ens[-L:]; p18_ens = p18_ens[-L:]; hi = hi[-L:]

    # 두 가지 submission: best_full, best_combined
    p_full = blend(p17_ens, p18_ens, hi, *best_combo_full[:3])
    p_comb = blend(p17_ens, p18_ens, hi, *best_combo_comb[:3])

    rows_out.append({"Bearing": nm, "N_meas": len(d["t_s"]),
                     "Last_t_s": float(d["t_s"][-1]),
                     "HI_last": float(hi[-1]),
                     "RUL_v17_s": float(p17_ens[-1]),
                     "RUL_v18_s": float(p18_ens[-1]),
                     "RUL_blend_full_s": float(p_full[-1]),
                     "RUL_blend_combined_s": float(p_comb[-1]),
                     "RUL_blend_full_h": float(p_full[-1]/3600),
                     "RUL_blend_combined_h": float(p_comb[-1]/3600)})
    print(f"  {nm}: HI_last={hi[-1]:.2f}  v17={p17_ens[-1]:.0f}s  v18={p18_ens[-1]:.0f}s  "
          f"blend_full={p_full[-1]:.0f}s  blend_combined={p_comb[-1]:.0f}s", flush=True)

out_df = pd.DataFrame(rows_out)
out_full = RESULT_DIR / "submission_v24_v17v22_full.xlsx"
out_comb = RESULT_DIR / "submission_v24_v17v22_combined.xlsx"
# 챌린지 포맷에 맞게 정리: RUL_pred_seconds, RUL_pred_hours
fmt_full = out_df[["Bearing","N_meas","Last_t_s","HI_last","RUL_blend_full_s","RUL_blend_full_h"]].copy()
fmt_full.columns = ["Bearing","N_measurements","Last_t_seconds","HI_last","RUL_pred_seconds","RUL_pred_hours"]
fmt_comb = out_df[["Bearing","N_meas","Last_t_s","HI_last","RUL_blend_combined_s","RUL_blend_combined_h"]].copy()
fmt_comb.columns = ["Bearing","N_measurements","Last_t_seconds","HI_last","RUL_pred_seconds","RUL_pred_hours"]
fmt_full.to_excel(out_full, index=False)
fmt_comb.to_excel(out_comb, index=False)
out_df.to_csv(RESULT_DIR / "submission_v24_v17v22_debug.csv", index=False)
print(f"\n  → {out_full}")
print(f"  → {out_comb}")
print(out_df.to_string(index=False))
