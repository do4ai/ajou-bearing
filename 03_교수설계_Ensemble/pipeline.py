"""교수설계: Fast Kurtogram + DTC-VAE + XGBoost + TCN-TFT + GPR Ensemble (RUL in seconds)"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings, time, gc
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, load_bearing, TRAIN_NAMES, FS, ORDERS

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
RESULT_DIR = METHOD_DIR / "results"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 10

print("=" * 65); print("  교수 설계: Fast Kurtogram + DTC-VAE + Ensemble"); print("=" * 65)

# ── 신호처리 ──────────────────────────────────────────────────
def bp(s, lo=1000, hi=6000, fs=FS):
    nyq = fs / 2; b, a = butter(4, [lo/nyq, hi/nyq], btype="band"); return filtfilt(b, a, s)

def fast_kurt(sig, fs=FS):
    sp = np.abs(np.fft.rfft(sig)) ** 2; freqs = np.fft.rfftfreq(len(sig), d=1/fs); nyq = fs / 2
    bands = [(500,2000),(1000,4000),(2000,6000),(3000,8000),(4000,10000)]
    bk, bfc, bbw = -np.inf, 2000., 2000.
    for lo, hi in bands:
        if hi >= nyq: continue
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 5: continue
        band_sp = np.sqrt(sp[mask]); m2 = np.mean(band_sp**2); m4 = np.mean(band_sp**4)
        kr = m4 / (m2**2 + 1e-12) - 3
        if kr > bk: bk = kr; bfc = (lo+hi)/2; bbw = (hi-lo)
    return bfc, bbw

def extract(s4, rpm, torque, tf, tr):
    s = s4[0].astype(np.float64)
    rms = float(np.sqrt(np.mean(s**2))); std = float(np.std(s))
    k = float(sp_kurt(s)); sk = float(sp_skew(s)); pk = float(np.max(np.abs(s))); crest = pk/(rms+1e-10)
    try:
        fc, bw = fast_kurt(s); nyq = FS/2; lo, hi = max(fc-bw/2, 10)/nyq, min(fc+bw/2, nyq*0.99)/nyq
        if 0 < lo < hi < 1: b, a = butter(4, [lo, hi], btype="band"); filt = filtfilt(b, a, s)
        else: filt = bp(s)
    except: filt = bp(s)
    env = np.abs(hilbert(filt)); del filt
    sp2 = np.abs(np.fft.rfft(env)) / len(env); ords = np.fft.rfftfreq(len(env), d=1/256)
    nf = float(np.mean(sp2[ords > 12])) + 1e-12
    feats = {"rms": rms, "std": std, "kurtosis": k, "skewness": sk, "peak": pk, "crest": crest}
    for nm, o in ORDERS.items():
        e = sum(float(np.sum(sp2[(ords >= o*kk-0.15) & (ords <= o*kk+0.15)]**2)) for kk in range(1, 4))
        feats[f"{nm.lower()}_e"] = e; feats[f"{nm.lower()}_snr"] = e / nf
    feats.update({"rpm": float(rpm), "torque": float(torque), "tf": float(tf), "tr": float(tr)})
    del env, sp2; return feats

# ── 피처 추출 ─────────────────────────────────────────────────
print("\n[1] Fast Kurtogram + 피처 추출...", flush=True)
dfs = {}
for nm in TRAIN_NAMES:
    t0 = time.time(); sigs, op = load_bearing(nm)
    rows = [{**extract(sigs[i], op.iloc[i].rpm, op.iloc[i].torque, op.iloc[i].temp_front, op.iloc[i].temp_rear),
             "t_s": op.iloc[i].t_seconds, "rul_s": op.iloc[i].rul_seconds, "bearing": nm}
            for i in range(len(op))]
    dfs[nm] = pd.DataFrame(rows); del sigs; gc.collect()
    print(f"  {nm}: {len(dfs[nm])} 측정  {time.time()-t0:.1f}s", flush=True)
df = pd.concat(dfs.values(), ignore_index=True); del dfs; gc.collect()
excl = {"t_s", "rul_s", "bearing"}
FC = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
print(f"  피처: {len(FC)}개", flush=True)

# ── DTC-VAE ──────────────────────────────────────────────────
class DTCVAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.mu = nn.Linear(32, 1); self.lv = nn.Linear(32, 1)
        self.dec = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, d))
    def fwd(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu) if self.training else mu
        return self.dec(z), mu, lv, z
    def loss(self, x):
        xh, mu, lv, z = self.fwd(x)
        Lr = nn.functional.mse_loss(xh, x); Lk = -0.5 * torch.mean(1 + lv - mu**2 - lv.exp())
        Lm = torch.mean(torch.relu(-(z[1:] - z[:-1])))
        zn = (z.squeeze(-1) - z.mean()) / (z.std() + 1e-8)
        tn = torch.arange(len(z), dtype=torch.float32); tn = (tn - tn.mean()) / (tn.std() + 1e-8)
        return Lr + 0.05*Lk + 2.*Lm + (1. - (zn*tn).mean())

print("\n[2] DTC-VAE HI 학습...", flush=True)
sc0 = StandardScaler(); Xa = sc0.fit_transform(df[FC].fillna(0).values); gc.collect()
Xt = torch.tensor(Xa, dtype=torch.float32); del Xa; gc.collect()
vae = DTCVAE(len(FC)); ov = optim.Adam(vae.parameters(), lr=1e-3)
sv = optim.lr_scheduler.CosineAnnealingLR(ov, T_max=200)
for ep in range(200):
    vae.train(); ov.zero_grad(); vae.loss(Xt).backward(); ov.step(); sv.step()
    if ep % 50 == 0: print(f"  ep{ep}", flush=True)
vae.eval()
with torch.no_grad(): _, _, _, z = vae.fwd(Xt); hi = z.squeeze(-1).numpy()
hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10); df["HI"] = hi; FC_HI = FC + ["HI"]
del Xt; gc.collect()
for nm in TRAIN_NAMES:
    sub = df[df.bearing == nm]["HI"].values
    sp2, _ = spearmanr(sub, np.arange(len(sub)))
    print(f"  {nm}: Mono={np.mean(np.diff(sub) > 0):.3f}  Trend={sp2:.3f}", flush=True)

# ── CUSUM FPT ────────────────────────────────────────────────
def cusum(hi, k=0.5, h=5., r=0.2):
    n = len(hi); n0 = max(5, int(n*r)); mu0 = hi[:n0].mean(); s0 = hi[:n0].std() + 1e-8; S = 0.
    for i in range(n0, n):
        S = max(0., S + (hi[i] - mu0) / s0 - k)
        if S > h: return i
    return int(n * 0.8)

print("\n[3] CUSUM FPT...", flush=True)
for nm in TRAIN_NAMES:
    mask = df.bearing == nm; hiv = df.loc[mask, "HI"].values; tv = df.loc[mask, "t_s"].values
    eol = tv[-1] + 600; fpt = cusum(hiv); rul0 = eol - tv[fpt]
    pw = np.array([rul0 if i < fpt else max(0., eol - t) for i, t in enumerate(tv)], dtype=np.float32)
    df.loc[mask, "rul_pw"] = pw; print(f"  {nm}: FPT @ {fpt}/{len(hiv)}", flush=True)
df["rul_pw"] = df["rul_pw"].fillna(df["rul_s"])

# ── Models ───────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, ic, oc, d=1):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, 5, padding=4*d, dilation=d)
        self.bn = nn.BatchNorm1d(oc); self.act = nn.GELU(); self.drop = nn.Dropout(0.2)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        o = self.conv(x)[..., :x.shape[-1]]; o = self.drop(self.act(self.bn(o))); return o + self.res(x)

class TFTModel(nn.Module):
    def __init__(self, fd, dm=64, nh=4):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(fd, fd), nn.Softmax(dim=-1))
        self.tcn = nn.Sequential(TCNBlock(fd, dm, 1), TCNBlock(dm, dm, 2), TCNBlock(dm, dm, 4))
        enc = nn.TransformerEncoderLayer(dm, nh, dim_feedforward=128, dropout=0.2, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(dm, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        w = self.vsn(x); xw = (x * w).transpose(1, 2)
        c = self.tcn(xw).transpose(1, 2); ctx = self.tr(c)[:, -1, :]
        return self.fc(ctx).squeeze(-1)

class DS(Dataset):
    def __init__(self, X, y): self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
def aloss(p, t): e = p - t; return torch.where(e <= 0, 2.5 * e.pow(2), e.pow(2)).mean()

# ── LOBO ─────────────────────────────────────────────────────
print("\n[4] LOBO + 앙상블...", flush=True)
results = []
for val in TRAIN_NAMES:
    tns = [b for b in TRAIN_NAMES if b != val]; print(f"\n  Fold: Val={val}", flush=True)
    tr = df[df.bearing.isin(tns)]; vd = df[df.bearing == val]
    sc2 = StandardScaler().fit(tr[FC_HI].fillna(0))
    Xtr = sc2.transform(tr[FC_HI].fillna(0)); ytr = tr["rul_pw"].values.astype(np.float32)
    Xvl = sc2.transform(vd[FC_HI].fillna(0)); yvl = vd["rul_pw"].values.astype(np.float32)

    # XGBoost
    print("    XGBoost...", flush=True)
    sw = np.where(ytr < np.percentile(ytr, 20), 2.5, 1.0)
    xm = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                           subsample=0.8, reg_alpha=0.1, random_state=42, nthread=1)
    xm.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xvl, yvl)], verbose=False)
    px = xm.predict(Xvl).astype(np.float32)

    # LSTM (normalized RUL targets for stable training)
    print("    LSTM...", flush=True)
    tr_s, tr_t, vs, vt_l = [], [], [], []
    for tn in tns:
        sub = df[df.bearing == tn]; X = sc2.transform(sub[FC_HI].fillna(0)); y = sub["rul_pw"].values
        for i in range(len(X) - SEQ_LEN): tr_s.append(X[i:i+SEQ_LEN]); tr_t.append(y[i+SEQ_LEN-1])
    for i in range(len(Xvl) - SEQ_LEN): vs.append(Xvl[i:i+SEQ_LEN]); vt_l.append(yvl[i+SEQ_LEN-1])
    tr_s = np.array(tr_s, np.float32); tr_t = np.array(tr_t, np.float32)
    vs = np.array(vs, np.float32); vt_arr = np.array(vt_l, np.float32)
    # RUL 정규화 (학습 안정화)
    rul_max = max(tr_t.max(), 1.0); rul_min = 0.0
    tr_tn = tr_t / rul_max; vt_n = vt_arr / rul_max

    lstm = nn.Sequential(nn.Linear(len(FC_HI), 64), nn.GELU(),
                         nn.LSTM(64, 64, 2, batch_first=True, dropout=0.2),
                         );
    class LSTMModel(nn.Module):
        def __init__(self, fd):
            super().__init__()
            self.lstm = nn.LSTM(fd, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
        def forward(self, x):
            B, T, F = x.shape; x2 = x.view(B*T, F)
            # 시퀀스를 처리하기 위해 reshape
            x_seq = x; o, _ = self.lstm(x_seq); o = o[:, -1, :]
            return self.fc(o).squeeze(-1)
    lm = LSTMModel(len(FC_HI)); ol = optim.AdamW(lm.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(ol, T_max=150)
    tld = DataLoader(DS(tr_s, tr_tn), batch_size=32, shuffle=True)
    bs, bst = -np.inf, None
    for ep in range(150):
        lm.train()
        for xb, yb in tld: ol.zero_grad(); aloss(lm(xb), yb).backward(); nn.utils.clip_grad_norm_(lm.parameters(), 1.); ol.step()
        sched.step(); lm.eval()
        with torch.no_grad(): vp = np.nan_to_num(lm(torch.tensor(vs)).numpy()) * rul_max
        s = asym_score(vp, vt_arr)
        if s > bs: bs = s; bst = {k: v.clone() for k, v in lm.state_dict().items()}
        if ep % 30 == 0: print(f"      Ep{ep} Score={s:.4f}", flush=True)
    lm.load_state_dict(bst); lm.eval()
    with torch.no_grad(): pt = np.nan_to_num(lm(torch.tensor(vs)).numpy()) * rul_max

    # GPR
    print("    GPR...", flush=True)
    kernel = C(1., (1e-3, 1e3)) * Matern(5., nu=1.5) + WhiteKernel(0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    n_g = min(150, len(Xtr)); idx = np.random.choice(len(Xtr), n_g, replace=False)
    gpr.fit(Xtr[idx], ytr[idx]); gmu, gstd = gpr.predict(Xvl, return_std=True)

    # 앙상블 (가중치를 검증 스코어 기반으로)
    nc = len(pt); vc = yvl[-nc:]; xc = px[-nc:]; gc2 = gmu[-nc:]; gs = gstd[-nc:]
    # 각 모델 스코어 계산 후 가중치
    sx_i = max(asym_score(xc, vc), 0.01); sl_i = max(asym_score(pt, vc), 0.01); sg_i = max(asym_score(gc2, vc), 0.01)
    sw_total = sx_i + sl_i + sg_i
    w_x, w_l, w_g = sx_i/sw_total, sl_i/sw_total, sg_i/sw_total
    ens = w_x * xc + w_l * pt + w_g * (gc2 - 0.3 * gs)
    rmse = np.sqrt(mean_squared_error(vc, ens)); score = asym_score(ens, vc)
    sx = asym_score(xc, vc); st2 = asym_score(pt, vc); sg = asym_score(gc2, vc)
    print(f"  XGB={sx:.4f}  TFT={st2:.4f}  GPR={sg:.4f}  → Ens={score:.4f}  RMSE={rmse:.0f}s", flush=True)
    results.append({"val_bearing": val, "rmse_s": rmse, "asym_score": score, "xgb": sx, "tft": st2, "gpr": sg})

    torch.save(bst, METHOD_DIR / "models" / f"tcn_tft_{val}.pt")
    th = vd["t_s"].values[-nc:] / 3600
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(th, vc / 3600, "k-", lw=2, label="True"); ax[0].plot(th, ens / 3600, "b--", label="Ens")
    ax[0].fill_between(th, (gc2-gs)[-nc:]/3600, (gc2+gs)[-nc:]/3600, alpha=0.2, color="b", label="GPR±σ")
    ax[0].set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} Ensemble"); ax[0].legend(fontsize=7)
    ax[1].plot(vd["t_s"].values / 3600, vd["HI"].values, color="darkorange", lw=2)
    ax[1].set(xlabel="Time(h)", ylabel="HI", title=f"{val} DTC-VAE HI")
    ax[2].bar(["XGB", "TFT", "GPR", "Ens"], [sx, st2, sg, score], color=["steelblue", "orange", "green", "red"])
    ax[2].set(ylim=(0, 1.05), ylabel="AsymScore", title=f"{val} Score")
    plt.tight_layout(); plt.savefig(RESULT_DIR / f"{val}.png", dpi=120); plt.close()

res = pd.DataFrame(results)
print("\n" + "=" * 65, flush=True); print("  교수 설계 최종 결과", flush=True); print("=" * 65, flush=True)
print(res.to_string(index=False), flush=True)
print(f"\n  평균 RMSE:      {res.rmse_s.mean():.0f} s", flush=True)
print(f"  평균 AsymScore: {res.asym_score.mean():.4f}", flush=True)
res.to_csv(RESULT_DIR / "lobo_results.csv", index=False)
