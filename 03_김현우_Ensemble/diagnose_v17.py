"""v17 진단: 진짜 평가지표(rul_s) vs 학습지표(rul_pw), 그리고 마지막 측정 점수.

v17 LOBO 0.6560은 **rul_pw 기준**. 실제 챌린지는 **rul_s 기준**으로 마지막 측정만!
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, load_bearing, TRAIN_NAMES, FS, ORDERS

import torch, torch.nn as nn
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
MODEL_DIR = METHOD_DIR / "models"
RESULT_DIR = METHOD_DIR / "results"

# ── 피처 추출 (pipeline.py 동일) ────────────────────────────────────
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

def extract(s4, rpm, torque, tf, tr):
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
    feats.update({"rpm": float(rpm), "torque": float(torque), "tf": float(tf),
                  "tr": float(tr), "power_proxy": float(rpm * abs(torque))})
    return feats


# ── 모델 아키텍처 ──────────────────────────────────────────────────
class DTCVAE(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU())
        self.mu = nn.Linear(32, latent); self.lv = nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.GELU(),
                                 nn.Linear(32, 64), nn.GELU(), nn.Linear(64, d))
    def fwd(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        return self.dec(mu), mu, lv, mu


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


# ── 피처 + HI 재구성 ───────────────────────────────────────────────
print("=" * 70); print("  v17 진단: rul_s vs rul_pw, full vs last-only"); print("=" * 70)

scaler_feats = joblib.load(MODEL_DIR / "scaler_features.pkl")
feat_meta = joblib.load(MODEL_DIR / "feature_meta.pkl")
FC = feat_meta["FC"]; FC_HI = feat_meta["FC_HI"]
vae = DTCVAE(len(FC), latent=4); vae.load_state_dict(torch.load(MODEL_DIR / "dtcvae.pt"))
vae.eval()

dfs = {}
for nm in TRAIN_NAMES:
    sigs, op = load_bearing(nm)
    rows = [{**extract(sigs[i], op.iloc[i].rpm, op.iloc[i].torque,
                       op.iloc[i].temp_front, op.iloc[i].temp_rear),
             "t_s": op.iloc[i].t_seconds, "rul_s": op.iloc[i].rul_seconds, "bearing": nm}
            for i in range(len(op))]
    dfs[nm] = pd.DataFrame(rows); del sigs
df = pd.concat(dfs.values(), ignore_index=True)

Xa = scaler_feats.transform(df[FC].fillna(0).values)
with torch.no_grad():
    _, _, _, z_all = vae.fwd(torch.tensor(Xa, dtype=torch.float32))
z_np = z_all.numpy()
hi = z_np[:, 0]; hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
df["HI"] = hi
for i in range(z_np.shape[1]): df[f"latent_{i}"] = z_np[:, i]

# Piecewise RUL
for nm in TRAIN_NAMES:
    mask = df.bearing == nm; hiv = df.loc[mask, "HI"].values; tv = df.loc[mask, "t_s"].values
    eol = tv[-1] + 600; fpt = cusum(hiv); rul0 = eol - tv[fpt]
    pw = np.array([rul0 if i < fpt else max(0., eol - t) for i, t in enumerate(tv)], dtype=np.float32)
    df.loc[mask, "rul_pw"] = pw
df["rul_pw"] = df["rul_pw"].fillna(df["rul_s"])


# ── 폴드별 예측 재생성 + 양쪽 평가지표 비교 ────────────────────────
print("\n[POSTPROCESS] 폴드별 OOF + 양쪽 지표 (rul_pw / rul_s / last-only)", flush=True)
oof_preds = {}

for val in TRAIN_NAMES:
    fold_dir = MODEL_DIR / f"fold_{val}"
    meta = joblib.load(fold_dir / "meta.pkl")
    sc2 = meta["scaler"]; rul_max = meta["rul_max"]
    SEQ_LEN = meta["SEQ_LEN"]
    TFT_SEEDS = meta["TFT_SEEDS"]; BILSTM_SEEDS = meta["BILSTM_SEEDS"]
    FC_HI_fold = meta["FC_HI"]

    vd = df[df.bearing == val].copy()
    Xvl_full = sc2.transform(vd[FC_HI_fold].fillna(0).values)
    yvl_pw = vd["rul_pw"].values.astype(np.float32)
    yvl_raw = vd["rul_s"].values.astype(np.float32)

    vs = np.array([Xvl_full[i:i+SEQ_LEN] for i in range(len(Xvl_full)-SEQ_LEN+1)], np.float32)
    vt_pw = yvl_pw[SEQ_LEN-1:]; vt_raw = yvl_raw[SEQ_LEN-1:]

    arch_seeds = [("tft", TFTModel, s) for s in TFT_SEEDS] + \
                 [("bilstm", BiLSTMModel, s) for s in BILSTM_SEEDS]
    fold_preds = []; fold_archs = []
    for arch_name, cls, sd in arch_seeds:
        model = cls(len(FC_HI_fold))
        ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
        model.load_state_dict(torch.load(ckpt)); model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(vs)).numpy() * rul_max
        fold_preds.append(raw); fold_archs.append(arch_name)
    fold_preds = np.array(fold_preds)

    gpr = joblib.load(fold_dir / "gpr.pkl")
    Xvl_seq_last = Xvl_full[SEQ_LEN-1:]
    _, gstd = gpr.predict(Xvl_seq_last, return_std=True)

    oof_preds[val] = dict(preds=fold_preds, archs=fold_archs, gstd=gstd,
                          yvl_raw=vt_raw, yvl_pw=vt_pw,
                          t_h=vd["t_s"].values[SEQ_LEN-1:] / 3600,
                          hi=vd["HI"].values[SEQ_LEN-1:])

# ── 전략별 점수 (rul_s vs rul_pw, full vs last) ───────────────────
def calc_strats(preds, gstd):
    pm = preds.mean(axis=0); pmed = np.median(preds, axis=0); pstd = preds.std(axis=0)
    sp_ = np.sort(preds, axis=0); ptrim = sp_[1:-1].mean(axis=0)
    pcons = pmed - 0.15*gstd - 0.10*pstd
    pultra = pmed - 0.30*gstd
    return dict(mean=pm, median=pmed, trim=ptrim, cons=pcons, ultra=pultra)

print("\n  ── Full trajectory ──")
print(f"  {'fold':10s} {'metric':12s} {'mean':8s} {'median':8s} {'trim':8s} {'cons':8s} {'ultra':8s}")
all_full_pw = {}; all_full_s = {}; all_last_pw = {}; all_last_s = {}
for val in TRAIN_NAMES:
    o = oof_preds[val]
    strats = calc_strats(o["preds"], o["gstd"])
    # Full traj
    s_pw = {k: asym_score(np.clip(v,0,None), o["yvl_pw"]) for k, v in strats.items()}
    s_s  = {k: asym_score(np.clip(v,0,None), o["yvl_raw"]) for k, v in strats.items()}
    # Last-only (1 point: last measurement)
    sl_pw = {k: asym_score(np.clip(v[-1:],0,None), o["yvl_pw"][-1:]) for k, v in strats.items()}
    sl_s  = {k: asym_score(np.clip(v[-1:],0,None), o["yvl_raw"][-1:]) for k, v in strats.items()}
    print(f"  {val:10s} {'full rul_pw':12s} " + "  ".join(f"{s_pw[k]:.4f}" for k in ["mean","median","trim","cons","ultra"]))
    print(f"  {val:10s} {'full rul_s ':12s} " + "  ".join(f"{s_s[k]:.4f}" for k in ["mean","median","trim","cons","ultra"]))
    print(f"  {val:10s} {'last rul_pw':12s} " + "  ".join(f"{sl_pw[k]:.4f}" for k in ["mean","median","trim","cons","ultra"]))
    print(f"  {val:10s} {'last rul_s ':12s} " + "  ".join(f"{sl_s[k]:.4f}" for k in ["mean","median","trim","cons","ultra"]))
    print(f"  {val:10s} → true RUL[last]={o['yvl_raw'][-1]:.0f}s  pred_median[last]={strats['median'][-1]:.0f}s",
          flush=True)
    all_full_pw[val] = s_pw; all_full_s[val] = s_s; all_last_pw[val] = sl_pw; all_last_s[val] = sl_s

print("\n  ── 4-fold 평균 ──")
for nm_metric, dd in [("full rul_pw", all_full_pw), ("full rul_s ", all_full_s),
                     ("last rul_pw", all_last_pw), ("last rul_s ", all_last_s)]:
    avg = {k: np.mean([dd[v][k] for v in TRAIN_NAMES]) for k in ["mean","median","trim","cons","ultra"]}
    print(f"  {nm_metric}:  " + "  ".join(f"{k}={avg[k]:.4f}" for k in avg))


# ── 시각화: 각 폴드별 trajectory ──────────────────────────────────
print("\n[VIZ] diagnose plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, val in zip(axes.flatten(), TRAIN_NAMES):
    o = oof_preds[val]
    strats = calc_strats(o["preds"], o["gstd"])
    th = o["t_h"]
    ax.plot(th, o["yvl_raw"]/3600, "k-", lw=2.5, label="TRUE rul_s")
    ax.plot(th, o["yvl_pw"]/3600, "k--", lw=1.5, alpha=0.6, label="rul_pw (학습타겟)")
    ax.plot(th, strats["median"]/3600, "b-", label=f"median")
    ax.plot(th, strats["cons"]/3600, "r--", label=f"cons")
    ax.plot(th, strats["ultra"]/3600, "purple", ls=":", label=f"ultra")
    ax.fill_between(th, (strats["mean"]-o["preds"].std(axis=0))/3600,
                    (strats["mean"]+o["preds"].std(axis=0))/3600, alpha=0.15, color="b")
    ax.set(xlabel="Time(h)", ylabel="RUL(h)",
           title=f"{val}: true_last={o['yvl_raw'][-1]:.0f}s  pred={strats['median'][-1]:.0f}s")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle("v17 진단 — 학습은 rul_pw, 채점은 rul_s, 제출은 last")
plt.tight_layout()
plt.savefig(RESULT_DIR / "diagnose_v17.png", dpi=120); plt.close()
print(f"  → {RESULT_DIR}/diagnose_v17.png")
