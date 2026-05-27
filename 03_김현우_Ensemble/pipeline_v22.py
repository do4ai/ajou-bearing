"""v22: v18 (EoL 50x) + 채널 대칭 피처 (v20에서 입증된 채널-invariant 신호)

v18의 약점: Train4 같은 channel-specific failure를 못 잡음 (ch0만 잘 보는 경향).
v20에서 채널 대칭 피처는 좋았지만 EoL 완화로 last 망함.
v22: v18 학습 설정 (EoL 50x, asym 5x) 그대로 + 채널 대칭 피처만 추가.

핵심 변경 (v18 → v22):
  1. 채널 대칭 피처 (max/min/range/std/top2 across 4 channels)
  2. 온도 차이 (tf-tr), temp_ratio
  3. 모든 학습 설정은 v18과 동일
"""
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
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
RESULT_DIR = METHOD_DIR / "results"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = METHOD_DIR / "models_v22"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 10
TFT_SEEDS = [42, 7, 123, 2026, 99]
BILSTM_SEEDS = [365, 1234, 777, 5050, 9999]
SEEDS = TFT_SEEDS + BILSTM_SEEDS
EPOCHS = 300
PATIENCE = 120
AUG_NOISE_STD = 0.035
TRIM = 3
SCORE_LOSS_START = 30
MIN_EPOCHS = 80
BILSTM_DM = 64
ASYM_PENALTY = 5.0           # v17: 2.5 → v18: 5.0 (늦은 예측 강력 페널티)
EOL_LAST_K = 5               # 마지막 5측정에 50x
EOL_MID_K = 15               # 다음 15측정에 10x
LATE_BIAS_INIT = 0.95        # 첫 예측에 0.95 곱하기 (선호적 early)

print("=" * 70); print("  v22: v18 + 채널 대칭 피처 (Train4 ch3-failure 대응)"); print("=" * 70)


# ── 4채널 신호처리 ─────────────────────────────────────────────────
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


def ch_feats(s, prefix):
    """단일 채널 기본 통계 + 주파수 피처."""
    s = s.astype(np.float64)
    rms = float(np.sqrt(np.mean(s**2))); std = float(np.std(s))
    k = float(sp_kurt(s)); sk = float(sp_skew(s))
    pk = float(np.max(np.abs(s))); crest = pk / (rms + 1e-10)
    p2p = float(np.ptp(s)); shape_f = rms / (np.mean(np.abs(s)) + 1e-10)
    try:
        fc, bw, sk_kurt = fast_kurt(s); nyq = FS/2
        lo, hi = max(fc-bw/2, 10)/nyq, min(fc+bw/2, nyq*0.99)/nyq
        if 0 < lo < hi < 1:
            b, a = butter(4, [lo, hi], btype="band"); filt = filtfilt(b, a, s)
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


def extract(s4, rpm, torque, tf, tr):
    """v22: 4채널 피처 + 채널 대칭 피처."""
    feats = {}
    per_ch = {}
    for ci in range(4):
        f = ch_feats(s4[ci], f"ch{ci}")
        feats.update(f); per_ch[ci] = f
    # ★ 채널 대칭 피처 (Train4 ch3-failure 같은 패턴 잡기) ★
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
    # 다채널 통합
    s_all = s4.astype(np.float64)
    feats["rms_multi"] = float(np.sqrt(np.mean(s_all**2)))
    feats["std_multi"] = float(np.std(s_all))
    feats["peak_multi"] = float(np.max(np.abs(s_all)))
    for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        c = np.corrcoef(s4[i].astype(np.float64), s4[j].astype(np.float64))[0,1]
        feats[f"corr_{i}{j}"] = float(c) if np.isfinite(c) else 0.0
    energies = np.array([float(np.mean(s4[i].astype(np.float64)**2)) for i in range(4)])
    feats["energy_max"] = float(energies.max()); feats["energy_min"] = float(energies.min())
    feats["energy_ratio"] = float(energies.max() / (energies.min() + 1e-10))
    feats["energy_std"] = float(energies.std())
    # 운전조건 + 온도 차이
    rpm_v = float(rpm) if (rpm == rpm) else 800.0
    tq_v = float(torque) if (torque == torque) else -5.0
    tf_v = float(tf) if (tf == tf) else 30.0
    tr_v = float(tr) if (tr == tr) else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v),
                  "temp_diff": tf_v - tr_v,
                  "temp_max": max(tf_v, tr_v),
                  "temp_ratio": tf_v / (tr_v + 1e-6)})
    return feats


print("\n[1] 4채널 피처 추출 (시간 좀 걸림: 4채널 × ~466 측정)...", flush=True)
dfs = {}
for nm in TRAIN_NAMES:
    t0 = time.time(); sigs, op = load_bearing(nm)
    rows = []
    for i in range(len(op)):
        row = op.iloc[i]
        f = extract(sigs[i], row.rpm, row.torque, row.temp_front, row.temp_rear)
        f["t_s"] = row.t_seconds; f["rul_s"] = row.rul_seconds; f["bearing"] = nm
        rows.append(f)
    dfs[nm] = pd.DataFrame(rows); del sigs; gc.collect()
    print(f"  {nm}: {len(dfs[nm])} meas × {len(rows[0])} feats  {time.time()-t0:.1f}s", flush=True)
df = pd.concat(dfs.values(), ignore_index=True)
excl = {"t_s", "rul_s", "bearing"}
FC = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
print(f"  총 피처: {len(FC)}개 (v17: 31 → v18: {len(FC)})", flush=True)


# ── DTC-VAE (동일하지만 input dim 다름) ───────────────────────────
class DTCVAE(nn.Module):
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
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu) if self.training else mu
        return self.dec(z), mu, lv, z
    def loss(self, x, bearing_ids):
        xh, mu, lv, z = self.fwd(x)
        Lr = nn.functional.mse_loss(xh, x); Lk = -0.5 * torch.mean(1 + lv - mu**2 - lv.exp())
        Lm = torch.tensor(0., device=x.device); Lt = torch.tensor(0., device=x.device)
        for bid in bearing_ids.unique():
            idx = (bearing_ids == bid).nonzero(as_tuple=True)[0]
            if len(idx) < 3: continue
            z_b = z[idx, 0]
            Lm = Lm + torch.mean(torch.relu(-(z_b[1:] - z_b[:-1])))
            zn = (z_b - z_b.mean()) / (z_b.std() + 1e-8)
            tn = torch.arange(len(z_b), dtype=torch.float32, device=x.device)
            tn = (tn - tn.mean()) / (tn.std() + 1e-8)
            Lt = Lt + (1. - (zn*tn).mean())
        return Lr + 0.05*Lk + 2.*Lm + 0.5*Lt


print("\n[2] DTC-VAE HI 학습...", flush=True)
torch.manual_seed(42); np.random.seed(42)
sc0 = StandardScaler(); Xa = sc0.fit_transform(df[FC].fillna(0).values)
Xt = torch.tensor(Xa, dtype=torch.float32); del Xa
bid_map = {nm: i for i, nm in enumerate(TRAIN_NAMES)}
bid_t = torch.tensor([bid_map[b] for b in df.bearing], dtype=torch.long)

vae = DTCVAE(len(FC), latent=4); ov = optim.Adam(vae.parameters(), lr=1e-3)
sv = optim.lr_scheduler.CosineAnnealingLR(ov, T_max=300)
for ep in range(300):
    vae.train(); ov.zero_grad(); vae.loss(Xt, bid_t).backward(); ov.step(); sv.step()
    if ep % 100 == 0: print(f"  ep{ep}", flush=True)
vae.eval()
with torch.no_grad(): _, _, _, z_all = vae.fwd(Xt)
z_np = z_all.numpy()
hi = z_np[:, 0]; hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
df["HI"] = hi
for i in range(z_np.shape[1]): df[f"latent_{i}"] = z_np[:, i]
FC_HI = FC + ["HI"] + [f"latent_{i}" for i in range(z_np.shape[1])]

print("  HI 품질:", flush=True)
for nm in TRAIN_NAMES:
    sub = df[df.bearing == nm]["HI"].values
    sp2, _ = spearmanr(sub, np.arange(len(sub)))
    print(f"    {nm}: Mono={np.mean(np.diff(sub) > 0):.3f}  Trend={sp2:.3f}", flush=True)

joblib.dump(sc0, MODEL_DIR / "scaler_features.pkl")
torch.save(vae.state_dict(), MODEL_DIR / "dtcvae.pt")
joblib.dump({"FC": FC, "FC_HI": FC_HI, "bid_map": bid_map}, MODEL_DIR / "feature_meta.pkl")


# ── CUSUM FPT (보조용) ──────────────────────────────────────────────
def cusum(hi, k=0.5, h=5., r=0.2):
    n = len(hi); n0 = max(5, int(n*r)); mu0 = hi[:n0].mean(); s0 = hi[:n0].std() + 1e-8; S = 0.
    for i in range(n0, n):
        S = max(0., S + (hi[i] - mu0) / s0 - k)
        if S > h: return i
    return int(n * 0.8)

print("\n[3] CUSUM FPT + Piecewise RUL (보조 task)...", flush=True)
fpt_dict = {}
for nm in TRAIN_NAMES:
    mask = df.bearing == nm; hiv = df.loc[mask, "HI"].values; tv = df.loc[mask, "t_s"].values
    eol = tv[-1] + 600; fpt = cusum(hiv); rul0 = eol - tv[fpt]
    pw = np.array([rul0 if i < fpt else max(0., eol - t) for i, t in enumerate(tv)], dtype=np.float32)
    df.loc[mask, "rul_pw"] = pw; fpt_dict[nm] = int(fpt)
    print(f"  {nm}: FPT @ {fpt}/{len(hiv)}  ({tv[fpt]/3600:.1f}h)", flush=True)
df["rul_pw"] = df["rul_pw"].fillna(df["rul_s"])


# ── Loss ───────────────────────────────────────────────────────────
def aloss_torch(p, t, penalty=ASYM_PENALTY):
    e = p - t
    return torch.where(e > 0, penalty * e.pow(2), e.pow(2)).mean()


def score_loss_torch(p_norm, t_norm, rul_max):
    p = p_norm * rul_max; t = t_norm * rul_max
    er = 100.0 * (t - p) / (t.abs() + 1.0)
    ln_half = float(np.log(0.5))
    arg_late = torch.clamp(-ln_half * (-er).clamp(min=0) / 20.0, -50.0, 0.0)
    arg_early = torch.clamp( ln_half *   er.clamp(min=0)  / 50.0, -50.0, 0.0)
    A = torch.where(er <= 0, arg_late.exp(), arg_early.exp())
    return 1.0 - A.mean()


# ── 모델 ───────────────────────────────────────────────────────────
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
    def __init__(self, fd, dm=BILSTM_DM):
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


# ── 데이터셋: EoL 가중치 50x ───────────────────────────────────────
class AugDS(Dataset):
    """v18: rul_s 직접 학습 + EoL 강조 가중치."""
    def __init__(self, X, y, eol_last_k=EOL_LAST_K, eol_mid_k=EOL_MID_K,
                 noise=AUG_NOISE_STD, mixup_alpha=0.2, mixup_prob=0.3):
        self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.float32)
        self.noise = noise; self.mixup_alpha = mixup_alpha; self.mixup_prob = mixup_prob
        # rul_s 기준 가중치: y 작을수록 EoL → 가중치 폭증
        # 마지막 EOL_LAST_K개에 50x, 다음 EOL_MID_K개에 10x, 그 외 1x
        y_np = self.y.numpy()
        order = np.argsort(y_np)  # 작은 RUL → 큰 인덱스부터 정렬됨
        w = np.ones_like(y_np, dtype=np.float32)
        # 정렬 후 작은 RUL k개에 가중치
        w[order[:eol_last_k]] = 50.0
        w[order[eol_last_k:eol_last_k+eol_mid_k]] = 10.0
        self.weight = torch.tensor(w, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]; y = self.y[i]; w = self.weight[i]
        if self.mixup_alpha > 0 and np.random.rand() < self.mixup_prob:
            j = np.random.randint(len(self))
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            x = lam * x + (1 - lam) * self.X[j]; y = lam * y + (1 - lam) * self.y[j]
            w = lam * w + (1 - lam) * self.weight[j]
        if self.noise > 0: x = x + torch.randn_like(x) * self.noise
        return x, y, w


def train_model(model_cls, Xtr_seq, ytr_seq_norm, Xvl_seq, yvl_raw, rul_max, seed,
                epochs=EPOCHS, patience=PATIENCE, score_start=None):
    if score_start is None: score_start = SCORE_LOSS_START
    torch.manual_seed(seed); np.random.seed(seed)
    fd = Xtr_seq.shape[-1]; model = model_cls(fd)
    ol = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(ol, T_0=80, T_mult=2)
    tld = DataLoader(AugDS(Xtr_seq, ytr_seq_norm), batch_size=32, shuffle=True)
    bs, bst, no_imp = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in tld:
            ol.zero_grad()
            # alpha: 60% asym MSE + 40% score loss (v17: 70/30)
            alpha = 1.0 if ep < score_start else max(0.30, 1.0 - (ep - score_start) / 100.0)
            pred = model(xb); e = pred - yb
            # ★ 가중 asym MSE: 늦은 예측 5x 페널티, EoL 50x 가중치 ★
            weighted_mse = (torch.where(e > 0, ASYM_PENALTY * e.pow(2), e.pow(2)) * wb).mean()
            score_l = score_loss_torch(pred, yb, rul_max)
            loss = alpha * weighted_mse + (1 - alpha) * score_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); ol.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vp = np.nan_to_num(model(torch.tensor(Xvl_seq)).numpy()) * rul_max
        vp = np.clip(vp, 600.0, None)  # ★ 600s 하한 ★
        # validation은 rul_s 기준 + last-only 점수 둘 다 추적
        s_full = asym_score(vp, yvl_raw)
        s_last = asym_score(vp[-1:], yvl_raw[-1:])
        # 0.7 * full + 0.3 * last 가중 결합으로 early stopping
        s = 0.7 * s_full + 0.3 * s_last
        if s > bs: bs = s; bst = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if ep >= MIN_EPOCHS and no_imp >= patience: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        pred = np.nan_to_num(model(torch.tensor(Xvl_seq)).numpy()) * rul_max
    pred = np.clip(pred, 600.0, None)
    return model, pred, bs


# ── LOBO ───────────────────────────────────────────────────────────
print(f"\n[4] LOBO + 학습 ({len(SEEDS)} seeds/fold, rul_s 직접)...", flush=True)
results = []

for val in TRAIN_NAMES:
    tns = [b for b in TRAIN_NAMES if b != val]
    print(f"\n  Fold: Val={val}", flush=True)
    tr = df[df.bearing.isin(tns)]; vd = df[df.bearing == val]
    sc2 = StandardScaler().fit(tr[FC_HI].fillna(0))
    Xtr_full = sc2.transform(tr[FC_HI].fillna(0)); ytr = tr["rul_s"].values.astype(np.float32)
    Xvl_full = sc2.transform(vd[FC_HI].fillna(0)); yvl = vd["rul_s"].values.astype(np.float32)

    # 베어링별 시퀀스
    tr_s, tr_t = [], []
    for tn in tns:
        sub = df[df.bearing == tn]
        X = sc2.transform(sub[FC_HI].fillna(0)); y = sub["rul_s"].values  # ★ rul_s ★
        for i in range(len(X) - SEQ_LEN + 1):
            tr_s.append(X[i:i+SEQ_LEN]); tr_t.append(y[i+SEQ_LEN-1])
    vs, vt_l = [], []
    for i in range(len(Xvl_full) - SEQ_LEN + 1):
        vs.append(Xvl_full[i:i+SEQ_LEN]); vt_l.append(yvl[i+SEQ_LEN-1])
    tr_s = np.array(tr_s, np.float32); tr_t = np.array(tr_t, np.float32)
    vs = np.array(vs, np.float32); vt_arr = np.array(vt_l, np.float32)
    rul_max = max(tr_t.max(), 1.0); tr_tn = tr_t / rul_max

    print(f"    TFT×{len(TFT_SEEDS)} + BiLSTM×{len(BILSTM_SEEDS)}", flush=True)
    fold_models = []; fold_preds = []; fold_best = []; fold_archs = []
    arch_list = [("TFT", TFTModel, TFT_SEEDS, 30), ("BiLSTM", BiLSTMModel, BILSTM_SEEDS, 10)]
    for arch_name, cls, sds, sc_start in arch_list:
        for sd in sds:
            t0 = time.time()
            model, pred, best = train_model(cls, tr_s, tr_tn, vs, vt_arr, rul_max, sd, score_start=sc_start)
            fold_models.append(model); fold_preds.append(pred); fold_best.append(best); fold_archs.append(arch_name)
            print(f"      {arch_name} seed={sd}: blended_best={best:.4f}  {time.time()-t0:.1f}s", flush=True)
    fold_preds = np.array(fold_preds)
    pred_mean = fold_preds.mean(axis=0)
    pred_median = np.median(fold_preds, axis=0)
    pred_std_seeds = fold_preds.std(axis=0)
    if len(fold_best) > 2 * TRIM:
        keep_idx = np.argsort(fold_best)[TRIM:]
        pred_trimmed = fold_preds[keep_idx].mean(axis=0)
    else:
        pred_trimmed = pred_mean

    # GPR σ
    kernel = (C(1., (1e-3, 1e3)) * Matern(10., nu=2.5)
              + C(0.5, (1e-3, 1e3)) * RBF(50.)
              + WhiteKernel(0.1, noise_level_bounds=(1e-5, 1.0)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                    alpha=0.05, normalize_y=True, random_state=42)
    n_g = min(200, len(Xtr_full)); idx = np.linspace(0, len(Xtr_full)-1, n_g).astype(int)
    gpr.fit(Xtr_full[idx], ytr[idx])
    Xvl_seq_last = Xvl_full[SEQ_LEN-1:]
    gmu, gstd = gpr.predict(Xvl_seq_last, return_std=True)

    p_med = np.clip(pred_median, 600, None)
    p_mean = np.clip(pred_mean, 600, None)
    p_cons = np.clip(pred_mean - 0.15 * gstd - 0.10 * pred_std_seeds, 600, None)
    p_ultra = np.clip(pred_mean - 0.30 * gstd, 600, None)
    p_trim = np.clip(pred_trimmed, 600, None)

    vc = vt_arr
    s_mean = asym_score(p_mean, vc); s_med = asym_score(p_med, vc)
    s_cons = asym_score(p_cons, vc); s_ultra = asym_score(p_ultra, vc)
    s_trim = asym_score(p_trim, vc)
    # last-only 점수 추가
    sl_mean = asym_score(p_mean[-1:], vc[-1:])
    sl_med = asym_score(p_med[-1:], vc[-1:])
    sl_trim = asym_score(p_trim[-1:], vc[-1:])
    candidates = {"mean": s_mean, "median": s_med, "cons": s_cons, "ultra": s_ultra, "trimmed": s_trim}
    best_strat = max(candidates, key=candidates.get); s_best = candidates[best_strat]

    rmse = np.sqrt(mean_squared_error(vc, p_mean))
    print(f"  Full: mean={s_mean:.4f}  med={s_med:.4f}  trim={s_trim:.4f}  cons={s_cons:.4f}  ultra={s_ultra:.4f}", flush=True)
    print(f"  Last: mean={sl_mean:.4f}  med={sl_med:.4f}  trim={sl_trim:.4f}  "
          f"true={vc[-1]:.0f}s  pred={p_med[-1]:.0f}s", flush=True)
    results.append({"val": val, "rmse_s": rmse,
                    "full_mean": s_mean, "full_med": s_med, "full_trim": s_trim,
                    "full_cons": s_cons, "full_ultra": s_ultra,
                    "last_mean": sl_mean, "last_med": sl_med, "last_trim": sl_trim,
                    "true_last": float(vc[-1]), "pred_last_med": float(p_med[-1]),
                    "best_strat": best_strat, "best_score": s_best,
                    "seeds": ",".join(f"{b:.3f}" for b in fold_best)})

    # 모델 저장
    fold_dir = MODEL_DIR / f"fold_{val}"; fold_dir.mkdir(exist_ok=True)
    for i, m in enumerate(fold_models):
        sd = SEEDS[i]; arch = fold_archs[i]
        torch.save(m.state_dict(), fold_dir / f"{arch.lower()}_seed{sd}.pt")
    joblib.dump(gpr, fold_dir / "gpr.pkl")
    joblib.dump({"scaler": sc2, "rul_max": rul_max, "FC_HI": FC_HI,
                 "TFT_SEEDS": TFT_SEEDS, "BILSTM_SEEDS": BILSTM_SEEDS,
                 "SEEDS": SEEDS, "SEQ_LEN": SEQ_LEN}, fold_dir / "meta.pkl")

    # 시각화
    th = vd["t_s"].values[SEQ_LEN-1:] / 3600
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(th, vc/3600, "k-", lw=2, label="True rul_s")
    ax[0].plot(th, p_mean/3600, "b-", label=f"mean ({s_mean:.3f})")
    ax[0].plot(th, p_med/3600, "g--", label=f"median ({s_med:.3f})")
    ax[0].plot(th, p_cons/3600, "r--", label=f"cons ({s_cons:.3f})")
    ax[0].fill_between(th, (pred_mean-pred_std_seeds)/3600,
                        (pred_mean+pred_std_seeds)/3600, alpha=0.2, color="b")
    ax[0].set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} v18 rul_s")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(vd["t_s"].values/3600, vd["HI"].values, color="darkorange", lw=2)
    ax[1].axvline(df.loc[df.bearing==val, "t_s"].iloc[fpt_dict[val]]/3600, ls="--", color="red", label="FPT")
    ax[1].set(xlabel="Time(h)", ylabel="HI", title=f"{val} HI"); ax[1].legend()
    ax[2].bar(["mean","med","trim","cons","ultra"], [s_mean, s_med, s_trim, s_cons, s_ultra],
              color=["blue","green","orange","red","purple"])
    ax[2].set(ylim=(0, 1.05), ylabel="AsymScore (full)", title=f"{val}")
    plt.tight_layout(); plt.savefig(RESULT_DIR / f"v22_{val}.png", dpi=120); plt.close()

res = pd.DataFrame(results)
print("\n" + "=" * 70, flush=True); print("  v22 최종 결과 (v18 + 채널 대칭)", flush=True); print("=" * 70, flush=True)
print(res.to_string(index=False), flush=True)
print(f"\n  Full mean:     {res.full_mean.mean():.4f}", flush=True)
print(f"  Full median:   {res.full_med.mean():.4f}", flush=True)
print(f"  Full trim:     {res.full_trim.mean():.4f}", flush=True)
print(f"  Full cons:     {res.full_cons.mean():.4f}", flush=True)
print(f"  Full ultra:    {res.full_ultra.mean():.4f}", flush=True)
print(f"  Last mean:     {res.last_mean.mean():.4f}", flush=True)
print(f"  Last median:   {res.last_med.mean():.4f}  ← 핵심 지표", flush=True)
print(f"  Last trim:     {res.last_trim.mean():.4f}", flush=True)
print(f"  RMSE 평균:     {res.rmse_s.mean():.0f}s", flush=True)
res.to_csv(RESULT_DIR / "lobo_v22.csv", index=False)
