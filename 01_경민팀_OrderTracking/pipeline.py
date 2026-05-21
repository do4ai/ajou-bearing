"""경민팀: Order Tracking + 16피처 + CNN-BiLSTM — 전체 파이프라인 (RUL in seconds)"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings, time
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, load_bearing, TRAIN_NAMES, FS, ORDERS

import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
RESULT_DIR = METHOD_DIR / "results"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS, LR, SEQ_LEN = 150, 1e-3, 10

print("=" * 65); print("  경민 팀: Order Tracking + CNN-BiLSTM"); print("=" * 65)

# ── 신호처리 ──────────────────────────────────────────────────
def bandpass(s, lo=1000, hi=6000, fs=FS):
    nyq = fs / 2; b, a = butter(4, [lo/nyq, hi/nyq], btype="band"); return filtfilt(b, a, s)
def angular_resample(s, rpm, spr=256, fs=FS):
    n = len(s); fr = rpm / 60; nr = fr * n / fs; no = max(int(nr * spr), 1)
    return np.interp(np.linspace(0, nr, no), np.linspace(0, nr, n), s).astype(np.float32)
def order_spec(env, spr=256):
    sp = np.abs(np.fft.rfft(env)) / len(env); return np.fft.rfftfreq(len(env), d=1/spr), sp
def band_e(ords, sp, t, bw=0.15, nh=3):
    return sum(float(np.sum(sp[(ords >= t*k-bw) & (ords <= t*k+bw)]**2)) for k in range(1, nh+1))

def extract(s4, rpm, torque, tf, tr):
    s = s4[0].astype(np.float64)
    rms = np.sqrt(np.mean(s**2)); std = np.std(s); k = sp_kurt(s); sk = sp_skew(s)
    pk = np.max(np.abs(s)); p2p = np.ptp(s); crest = pk / (rms + 1e-10)
    feats = {"rms": float(rms), "std": float(std), "kurtosis": float(k), "skewness": float(sk),
             "peak": float(pk), "p2p": float(p2p), "crest": float(crest),
             "shape": float(rms / (np.mean(np.abs(s)) + 1e-10))}
    filt = bandpass(s); res = angular_resample(filt, rpm)
    env = np.abs(hilbert(res)); ords, sp = order_spec(env)
    nf = float(np.mean(sp[ords > 12])) + 1e-12
    for nm, o in ORDERS.items():
        e = band_e(ords, sp, o); feats[f"{nm.lower()}_e"] = e; feats[f"{nm.lower()}_snr"] = e / nf
    te = float(np.sum(sp**2)) + 1e-12
    feats["fault_ratio"] = sum(feats[f"{k.lower()}_e"] for k in ORDERS) / te
    feats.update({"rpm": float(rpm), "torque": float(torque), "tf": float(tf), "tr": float(tr)})
    return feats

# ── 피처 추출 ─────────────────────────────────────────────────
print("\n[1] 피처 추출...")
dfs = {}
for nm in TRAIN_NAMES:
    t0 = time.time(); sigs, op = load_bearing(nm)
    rows = [{**extract(sigs[i], op.iloc[i].rpm, op.iloc[i].torque, op.iloc[i].temp_front, op.iloc[i].temp_rear),
             "t_s": op.iloc[i].t_seconds, "rul_s": op.iloc[i].rul_seconds, "bearing": nm}
            for i in range(len(op))]
    dfs[nm] = pd.DataFrame(rows); print(f"  {nm}: {len(dfs[nm])} 측정  {time.time()-t0:.1f}s")
df = pd.concat(dfs.values(), ignore_index=True)
excl = {"t_s", "rul_s", "bearing"}
FC = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
print(f"  피처: {len(FC)}개")

# ── Health Indicator ──────────────────────────────────────────
print("\n[2] Health Indicator...")
sc_scores = {}
for col in FC:
    ms, ts = [], []
    for nm in TRAIN_NAMES:
        v = df[df.bearing == nm][col].values
        ms.append(float(np.mean(np.diff(v) > 0)))
        ts.append(abs(float(np.corrcoef(v, np.arange(len(v)))[0, 1])) if np.std(v) > 1e-10 else 0.)
    sc_scores[col] = (np.mean(ms) + np.mean(ts)) / 2
warr = np.array([sc_scores[c] for c in FC]); wts = np.exp(warr) / np.sum(np.exp(warr))
scl = StandardScaler(); Xn = scl.fit_transform(df[FC].fillna(0).values)
hi = Xn @ wts; hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10); df["HI"] = hi
FC_HI = FC + ["HI"]

# ── 모델 ──────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, X, y): self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class Model(nn.Module):
    def __init__(self, fd):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(fd, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
                                 nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2))
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softmax(dim=1))
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        c = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1); l, _ = self.lstm(c)
        w = self.attn(l); ctx = (l * w).sum(1); return self.fc(ctx).squeeze(-1)

def aloss(p, t):
    e = p - t
    return torch.where(e <= 0, 2.5 * e.pow(2), e.pow(2)).mean()

# ── LOBO ──────────────────────────────────────────────────────
print("\n[3] LOBO + CNN-BiLSTM...")
results = []
for val in TRAIN_NAMES:
    tns = [b for b in TRAIN_NAMES if b != val]; print(f"\n  Fold: Val={val}")
    tr = df[df.bearing.isin(tns)]; vl = df[df.bearing == val]
    sc2 = StandardScaler().fit(tr[FC_HI].fillna(0))
    seqs, tgts = [], []
    for tn in tns:
        sub = df[df.bearing == tn]; X = sc2.transform(sub[FC_HI].fillna(0)); y = sub["rul_s"].values
        for i in range(len(X) - SEQ_LEN): seqs.append(X[i:i+SEQ_LEN]); tgts.append(y[i+SEQ_LEN-1])
    seqs = np.array(seqs, dtype=np.float32); tgts = np.array(tgts, dtype=np.float32)
    Xv = sc2.transform(vl[FC_HI].fillna(0)); yv = vl["rul_s"].values
    vs, vt = [], []
    for i in range(len(Xv) - SEQ_LEN): vs.append(Xv[i:i+SEQ_LEN]); vt.append(yv[i+SEQ_LEN-1])
    vs = np.array(vs, dtype=np.float32); vt = np.array(vt, dtype=np.float32)
    # RUL 정규화 (학습 안정화)
    rul_max = max(tgts.max(), 1.0); tgts_n = tgts / rul_max; vt_n = vt / rul_max

    m = Model(len(FC_HI)); opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tld = DataLoader(DS(seqs, tgts_n), batch_size=32, shuffle=True)
    vld = DataLoader(DS(vs, vt_n), batch_size=64)
    best_sc, best_st = -np.inf, None
    for ep in range(1, EPOCHS+1):
        m.train()
        for xb, yb in tld:
            opt.zero_grad(); aloss(m(xb), yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad(): vp = np.nan_to_num(np.concatenate([m(xb).numpy() for xb, _ in vld])) * rul_max
        s = asym_score(vp, vt)
        if s > best_sc: best_sc = s; best_st = {k: v.clone() for k, v in m.state_dict().items()}
        if ep % 25 == 0: print(f"    Ep{ep:3d} Score={s:.4f} RMSE={np.sqrt(mean_squared_error(vt, vp)):.0f}s")

    m.load_state_dict(best_st); m.eval()
    with torch.no_grad(): preds = np.nan_to_num(np.concatenate([m(xb).numpy() for xb, _ in vld])) * rul_max
    rmse = np.sqrt(mean_squared_error(vt, preds)); score = asym_score(preds, vt)
    results.append({"val_bearing": val, "rmse_s": rmse, "asym_score": score})
    print(f"  → RMSE={rmse:.0f}s  AsymScore={score:.4f}")

    # 저장
    torch.save(best_st, METHOD_DIR / "models" / f"model_{val}.pt")
    th = vl["t_s"].values[SEQ_LEN:] / 3600
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(th, vt / 3600, "k-", lw=2, label="True"); ax[0].plot(th, preds / 3600, "b--", label="Pred")
    ax[0].set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} CNN-BiLSTM"); ax[0].legend()
    ax[1].plot(vl["t_s"].values / 3600, vl["HI"].values, color="darkorange", lw=2)
    ax[1].set(xlabel="Time(h)", ylabel="HI", title=f"{val} HI")
    plt.tight_layout(); plt.savefig(RESULT_DIR / f"{val}.png", dpi=120); plt.close()

res = pd.DataFrame(results)
print("\n" + "=" * 65); print("  경민 팀 최종 결과"); print("=" * 65)
print(res.to_string(index=False))
print(f"\n  평균 RMSE:      {res.rmse_s.mean():.0f} s")
print(f"  평균 AsymScore: {res.asym_score.mean():.4f}")
res.to_csv(RESULT_DIR / "lobo_results.csv", index=False)
