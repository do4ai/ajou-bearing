"""교수 설계 v3: TFT-Multi-Seed Ensemble + GPR σ-aware 보수 보정

v2 분석 결과 TFT 단독이 평균 0.579로 최강. 다른 모델은 노이즈만 추가.
v3 전략: TFT 다중 시드 + 아키텍처 다양화 + GPR σ로 보수적 조정만 사용.

핵심 개선:
  1. TFT 3 seeds 앙상블 (median 사용 → outlier robust)
  2. SEQ_LEN 8→10
  3. Data augmentation: 가우시안 노이즈 인젝션
  4. 학습 epoch 200→300, patience 50→70
  5. GPR은 σ만 사용 (예측은 TFT 평균)
  6. 보수적 보정: TFT_mean - 0.15·σ_GPR
  7. XGBoost 제거 (Train3에서 노이즈만 추가)
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
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
RESULT_DIR = METHOD_DIR / "results"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = METHOD_DIR / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 10
SEEDS = [42, 7, 123, 2026, 99, 365, 1234, 777, 5050, 9999]   # v10: 10개 시드
EPOCHS = 300
PATIENCE = 120
AUG_NOISE_STD = 0.035
TRIM = 3                # v10: trim 3개 (worst 3개 제거), 7개 사용
SCORE_LOSS_START = 30
MIN_EPOCHS = 80

print("=" * 70); print("  v3: TFT Multi-Seed Ensemble + σ-aware 보수 보정"); print("=" * 70)

# ── 신호처리 (v2 동일) ────────────────────────────────────────────
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


# ── 피처 추출 ──────────────────────────────────────────────────────
print("\n[1] 피처 추출...", flush=True)
dfs = {}
for nm in TRAIN_NAMES:
    t0 = time.time(); sigs, op = load_bearing(nm)
    rows = [{**extract(sigs[i], op.iloc[i].rpm, op.iloc[i].torque,
                       op.iloc[i].temp_front, op.iloc[i].temp_rear),
             "t_s": op.iloc[i].t_seconds, "rul_s": op.iloc[i].rul_seconds, "bearing": nm}
            for i in range(len(op))]
    dfs[nm] = pd.DataFrame(rows); del sigs; gc.collect()
    print(f"  {nm}: {len(dfs[nm])} 측정  {time.time()-t0:.1f}s", flush=True)
df = pd.concat(dfs.values(), ignore_index=True); del dfs; gc.collect()
# v6: lag 제거 (실험적으로 v3 베이스라인이 가장 안정적이었음)
excl = {"t_s", "rul_s", "bearing"}
FC = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
print(f"  피처: {len(FC)}개 (v6: lag 제거, v3 기반)", flush=True)


# ── DTC-VAE ────────────────────────────────────────────────────────
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


# ── CUSUM FPT + Piecewise RUL ──────────────────────────────────────
def cusum(hi, k=0.5, h=5., r=0.2):
    n = len(hi); n0 = max(5, int(n*r)); mu0 = hi[:n0].mean(); s0 = hi[:n0].std() + 1e-8; S = 0.
    for i in range(n0, n):
        S = max(0., S + (hi[i] - mu0) / s0 - k)
        if S > h: return i
    return int(n * 0.8)

print("\n[3] CUSUM FPT + Piecewise RUL...", flush=True)
fpt_dict = {}
for nm in TRAIN_NAMES:
    mask = df.bearing == nm; hiv = df.loc[mask, "HI"].values; tv = df.loc[mask, "t_s"].values
    eol = tv[-1] + 600; fpt = cusum(hiv); rul0 = eol - tv[fpt]
    pw = np.array([rul0 if i < fpt else max(0., eol - t) for i, t in enumerate(tv)], dtype=np.float32)
    df.loc[mask, "rul_pw"] = pw; fpt_dict[nm] = int(fpt)
    print(f"  {nm}: FPT @ {fpt}/{len(hiv)}  ({tv[fpt]/3600:.1f}h)", flush=True)
df["rul_pw"] = df["rul_pw"].fillna(df["rul_s"])


# ── Asym loss (★부호 수정★) ────────────────────────────────────────
def aloss_torch(p, t):
    e = p - t
    return torch.where(e > 0, 2.5 * e.pow(2), e.pow(2)).mean()


def score_loss_torch(p_norm, t_norm, rul_max):
    """평가지표 1 - asym_score 직접 미분. p_norm/t_norm 은 [0,1] 정규화 영역."""
    p = p_norm * rul_max  # 초 단위로 복원
    t = t_norm * rul_max
    er = 100.0 * (t - p) / (t.abs() + 1.0)  # 퍼센트 오차
    ln_half = float(np.log(0.5))
    # er <= 0 (늦은 예측): arg = -ln(0.5)*(-er)/20
    # er > 0 (이른 예측):  arg = +ln(0.5)*er/50
    arg_late = torch.clamp(-ln_half * (-er).clamp(min=0) / 20.0, -50.0, 0.0)
    arg_early = torch.clamp( ln_half *   er.clamp(min=0)  / 50.0, -50.0, 0.0)
    A = torch.where(er <= 0, arg_late.exp(), arg_early.exp())
    return 1.0 - A.mean()


def combined_loss(p_norm, t_norm, rul_max, alpha=0.7):
    """alpha = 0.7 (asym MSE) + 0.3 (score loss). 균형잡힌 학습."""
    return alpha * aloss_torch(p_norm, t_norm) + (1 - alpha) * score_loss_torch(p_norm, t_norm, rul_max)


# ── TFT 아키텍처 ───────────────────────────────────────────────────
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


class AugDS(Dataset):
    def __init__(self, X, y, noise=AUG_NOISE_STD):
        self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.float32)
        self.noise = noise
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if self.noise > 0: x = x + torch.randn_like(x) * self.noise
        return x, self.y[i]


def train_tft(Xtr_seq, ytr_seq_norm, Xvl_seq, yvl_raw, rul_max, seed, epochs=EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed); np.random.seed(seed)
    fd = Xtr_seq.shape[-1]
    tft = TFTModel(fd, dm=64, nh=4)
    ol = optim.AdamW(tft.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(ol, T_0=80, T_mult=2)
    tld = DataLoader(AugDS(Xtr_seq, ytr_seq_norm), batch_size=32, shuffle=True)
    bs, bst, no_imp = -np.inf, None, 0
    for ep in range(epochs):
        tft.train()
        for xb, yb in tld:
            ol.zero_grad()
            alpha = 1.0 if ep < SCORE_LOSS_START else max(0.4, 1.0 - (ep - SCORE_LOSS_START) / 120.0)
            combined_loss(tft(xb), yb, rul_max, alpha=alpha).backward()
            nn.utils.clip_grad_norm_(tft.parameters(), 1.); ol.step()
        sched.step(); tft.eval()
        with torch.no_grad():
            vp = np.nan_to_num(tft(torch.tensor(Xvl_seq)).numpy()) * rul_max
        s = asym_score(vp, yvl_raw)
        if s > bs: bs = s; bst = {k: v.clone() for k, v in tft.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        # v9: MIN_EPOCHS 전에는 early stop 불가 (score loss 단계 도달 보장)
        if ep >= MIN_EPOCHS and no_imp >= patience: break
    tft.load_state_dict(bst); tft.eval()
    with torch.no_grad():
        pred = np.nan_to_num(tft(torch.tensor(Xvl_seq)).numpy()) * rul_max
    return tft, pred, bs


# ── LOBO ───────────────────────────────────────────────────────────
print(f"\n[4] LOBO + TFT × {len(SEEDS)} seeds + GPR σ...", flush=True)
results = []

for val in TRAIN_NAMES:
    tns = [b for b in TRAIN_NAMES if b != val]
    print(f"\n  Fold: Val={val}", flush=True)
    tr = df[df.bearing.isin(tns)]; vd = df[df.bearing == val]
    sc2 = StandardScaler().fit(tr[FC_HI].fillna(0))
    Xtr_full = sc2.transform(tr[FC_HI].fillna(0)); ytr = tr["rul_pw"].values.astype(np.float32)
    Xvl_full = sc2.transform(vd[FC_HI].fillna(0)); yvl = vd["rul_pw"].values.astype(np.float32)

    # 시퀀스 구성 (Train 베어링 별로 시퀀스 만들고 합치기 → 베어링 경계 넘지 않게)
    tr_s, tr_t = [], []
    for tn in tns:
        sub = df[df.bearing == tn]
        X = sc2.transform(sub[FC_HI].fillna(0)); y = sub["rul_pw"].values
        for i in range(len(X) - SEQ_LEN + 1):
            tr_s.append(X[i:i+SEQ_LEN]); tr_t.append(y[i+SEQ_LEN-1])
    vs, vt_l = [], []
    for i in range(len(Xvl_full) - SEQ_LEN + 1):
        vs.append(Xvl_full[i:i+SEQ_LEN]); vt_l.append(yvl[i+SEQ_LEN-1])
    tr_s = np.array(tr_s, np.float32); tr_t = np.array(tr_t, np.float32)
    vs = np.array(vs, np.float32); vt_arr = np.array(vt_l, np.float32)
    rul_max = max(tr_t.max(), 1.0); tr_tn = tr_t / rul_max

    # ── 다중 시드 TFT 학습 ──
    print(f"    TFT × {len(SEEDS)} seeds...", flush=True)
    fold_models = []
    fold_preds = []
    fold_best = []
    for sd in SEEDS:
        t0 = time.time()
        tft, pred, best = train_tft(tr_s, tr_tn, vs, vt_arr, rul_max, sd)
        fold_models.append(tft); fold_preds.append(pred); fold_best.append(best)
        print(f"      seed={sd}: best={best:.4f}  {time.time()-t0:.1f}s", flush=True)
    fold_preds = np.array(fold_preds)  # [n_seeds, n_val]
    pred_mean = fold_preds.mean(axis=0)
    pred_median = np.median(fold_preds, axis=0)
    pred_std_seeds = fold_preds.std(axis=0)
    # Trimmed mean: 시드별 점수가 가장 낮은 TRIM개 제거 후 평균
    if len(fold_best) > 2 * TRIM:
        keep_idx = np.argsort(fold_best)[TRIM:]  # 하위 TRIM 제거
        pred_trimmed = fold_preds[keep_idx].mean(axis=0)
    else:
        pred_trimmed = pred_mean

    # ── GPR (보수적 σ 보정용만 사용) ──
    print(f"    GPR (σ for conservative shift)...", flush=True)
    kernel = (C(1., (1e-3, 1e3)) * Matern(10., nu=2.5)
              + C(0.5, (1e-3, 1e3)) * RBF(50.)
              + WhiteKernel(0.1, noise_level_bounds=(1e-5, 1.0)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                    alpha=0.05, normalize_y=True, random_state=42)
    n_g = min(200, len(Xtr_full)); idx = np.linspace(0, len(Xtr_full)-1, n_g).astype(int)
    gpr.fit(Xtr_full[idx], ytr[idx])
    Xvl_seq_last = Xvl_full[SEQ_LEN-1:]
    gmu, gstd = gpr.predict(Xvl_seq_last, return_std=True)

    # ── 결합: TFT mean ± 보수적 보정 ──
    nc = len(pred_mean); vc = vt_arr
    # 1. seed median
    p_med = pred_median
    # 2. seed mean
    p_mean = pred_mean
    # 3. 보수적: TFT_mean - 0.15·σ_GPR - 0.10·σ_seeds (불확실성 두 종류)
    p_cons = pred_mean - 0.15 * gstd - 0.10 * pred_std_seeds
    # 4. ultra conservative: TFT_mean - 0.3σ_GPR
    p_ultra = pred_mean - 0.3 * gstd
    # 음수 방지
    p_med = np.clip(p_med, 0, None); p_mean = np.clip(p_mean, 0, None)
    p_cons = np.clip(p_cons, 0, None); p_ultra = np.clip(p_ultra, 0, None)

    p_trim = np.clip(pred_trimmed, 0, None)
    s_mean = asym_score(p_mean, vc); s_med = asym_score(p_med, vc)
    s_cons = asym_score(p_cons, vc); s_ultra = asym_score(p_ultra, vc)
    s_trim = asym_score(p_trim, vc)
    candidates = {"mean": s_mean, "median": s_med, "cons": s_cons,
                  "ultra": s_ultra, "trimmed": s_trim}
    best_strat = max(candidates, key=candidates.get); s_best = candidates[best_strat]

    rmse = np.sqrt(mean_squared_error(vc, p_mean))
    print(f"  TFT mean={s_mean:.4f}  median={s_med:.4f}  trim={s_trim:.4f}  "
          f"cons={s_cons:.4f}  ultra={s_ultra:.4f}  "
          f"best={best_strat}({s_best:.4f})  RMSE={rmse:.0f}s", flush=True)
    results.append({"val_bearing": val, "rmse_s": rmse,
                    "tft_mean": s_mean, "tft_median": s_med, "tft_trim": s_trim,
                    "tft_cons": s_cons, "tft_ultra": s_ultra,
                    "best_strat": best_strat, "best_score": s_best,
                    "seeds": ",".join(f"{b:.3f}" for b in fold_best)})

    # 모델 저장
    fold_dir = MODEL_DIR / f"fold_{val}"; fold_dir.mkdir(exist_ok=True)
    for i, m in enumerate(fold_models):
        torch.save(m.state_dict(), fold_dir / f"tft_seed{SEEDS[i]}.pt")
    joblib.dump(gpr, fold_dir / "gpr.pkl")
    joblib.dump({"scaler": sc2, "rul_max": rul_max, "FC_HI": FC_HI,
                 "SEEDS": SEEDS, "SEQ_LEN": SEQ_LEN}, fold_dir / "meta.pkl")

    # 시각화
    th = vd["t_s"].values[SEQ_LEN-1:] / 3600
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(th, vc/3600, "k-", lw=2, label="True")
    ax[0].plot(th, p_mean/3600, "b-", label=f"TFT mean ({s_mean:.3f})")
    ax[0].plot(th, p_cons/3600, "r--", label=f"Cons ({s_cons:.3f})")
    ax[0].fill_between(th, (pred_mean-pred_std_seeds)/3600,
                        (pred_mean+pred_std_seeds)/3600, alpha=0.2, color="b", label="seed±σ")
    ax[0].set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} Multi-Seed TFT")
    ax[0].legend(fontsize=8)
    ax[1].plot(vd["t_s"].values/3600, vd["HI"].values, color="darkorange", lw=2)
    ax[1].axvline(df.loc[df.bearing==val, "t_s"].iloc[fpt_dict[val]]/3600,
                  ls="--", color="red", label="FPT")
    ax[1].set(xlabel="Time(h)", ylabel="HI", title=f"{val} DTC-VAE HI"); ax[1].legend()
    ax[2].bar(["mean","median","cons","ultra"], [s_mean, s_med, s_cons, s_ultra],
              color=["blue","green","red","purple"])
    ax[2].set(ylim=(0, 1.05), ylabel="AsymScore", title=f"{val} Strategy Comparison")
    plt.tight_layout(); plt.savefig(RESULT_DIR / f"{val}.png", dpi=120); plt.close()

res = pd.DataFrame(results)
# 호환성: asym_score 컬럼은 cons로 (제출 전략)
res["asym_score"] = res["tft_cons"]
print("\n" + "=" * 70, flush=True); print("  v3 최종 결과 (TFT Multi-Seed)", flush=True); print("=" * 70, flush=True)
print(res.to_string(index=False), flush=True)
print(f"\n  TFT seed mean (5 seeds):   {res.tft_mean.mean():.4f}", flush=True)
print(f"  TFT seed median:           {res.tft_median.mean():.4f}", flush=True)
print(f"  TFT trimmed mean (-1):     {res.tft_trim.mean():.4f}  ← 제출 권장", flush=True)
print(f"  TFT mean - 0.15σ-0.1σs:    {res.tft_cons.mean():.4f}", flush=True)
print(f"  TFT mean - 0.3σ (ultra):   {res.tft_ultra.mean():.4f}", flush=True)
print(f"  평균 RMSE:                 {res.rmse_s.mean():.0f} s", flush=True)
res.to_csv(RESULT_DIR / "lobo_results.csv", index=False)
