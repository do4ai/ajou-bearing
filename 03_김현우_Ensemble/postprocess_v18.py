"""v18 후처리: 저장된 v17 모델 그대로 사용 → 폴드별 최적 전략 + asym-aware β 보정.

목표:
  1. 폴드별 LOBO 예측 재생성 (저장 모델 로드)
  2. 모든 전략 (mean/median/trim/cons/ultra) 비교
  3. asym-aware β 그리드서치 (0.80 ~ 1.05)로 추가 보정
  4. 폴드별 HI 곡선 특성 → 자동 전략 선택기 후보 도출
  5. 결과를 lobo_results_v18.csv 로 저장
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
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew, spearmanr
import joblib

METHOD_DIR = Path(__file__).resolve().parent
MODEL_DIR = METHOD_DIR / "models"
RESULT_DIR = METHOD_DIR / "results"

# ── pipeline.py 와 동일한 피처 추출 ────────────────────────────────
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


# ── 모델 아키텍처 (pipeline.py 와 동일) ────────────────────────────
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
        z = mu  # eval 모드
        return self.dec(z), mu, lv, z


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
print("=" * 70); print("  v18 후처리: 폴드별 최적 전략 + asym-aware β"); print("=" * 70)

print("\n[1] 피처 + HI 재구성 (저장된 VAE 사용)...", flush=True)
scaler_feats = joblib.load(MODEL_DIR / "scaler_features.pkl")
feat_meta = joblib.load(MODEL_DIR / "feature_meta.pkl")
FC = feat_meta["FC"]; FC_HI = feat_meta["FC_HI"]
vae = DTCVAE(len(FC), latent=4); vae.load_state_dict(torch.load(MODEL_DIR / "dtcvae.pt"))
vae.eval()

dfs = {}
import time
for nm in TRAIN_NAMES:
    t0 = time.time(); sigs, op = load_bearing(nm)
    rows = [{**extract(sigs[i], op.iloc[i].rpm, op.iloc[i].torque,
                       op.iloc[i].temp_front, op.iloc[i].temp_rear),
             "t_s": op.iloc[i].t_seconds, "rul_s": op.iloc[i].rul_seconds, "bearing": nm}
            for i in range(len(op))]
    dfs[nm] = pd.DataFrame(rows); del sigs
    print(f"  {nm}: {len(dfs[nm])} 측정  {time.time()-t0:.1f}s", flush=True)
df = pd.concat(dfs.values(), ignore_index=True)

# HI 재구성
Xa = scaler_feats.transform(df[FC].fillna(0).values)
with torch.no_grad():
    _, _, _, z_all = vae.fwd(torch.tensor(Xa, dtype=torch.float32))
z_np = z_all.numpy()
hi = z_np[:, 0]; hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
df["HI"] = hi
for i in range(z_np.shape[1]): df[f"latent_{i}"] = z_np[:, i]

# Piecewise RUL 재구성
print("\n[2] CUSUM FPT + Piecewise RUL...", flush=True)
fpt_dict = {}
for nm in TRAIN_NAMES:
    mask = df.bearing == nm; hiv = df.loc[mask, "HI"].values; tv = df.loc[mask, "t_s"].values
    eol = tv[-1] + 600; fpt = cusum(hiv); rul0 = eol - tv[fpt]
    pw = np.array([rul0 if i < fpt else max(0., eol - t) for i, t in enumerate(tv)], dtype=np.float32)
    df.loc[mask, "rul_pw"] = pw; fpt_dict[nm] = int(fpt)
df["rul_pw"] = df["rul_pw"].fillna(df["rul_s"])


# ── 폴드별 예측 재생성 ─────────────────────────────────────────────
print(f"\n[3] 폴드별 OOF 예측 재생성...", flush=True)
results = []
oof_preds = {}  # {fold: [model_preds, gpr_std]}

for val in TRAIN_NAMES:
    fold_dir = MODEL_DIR / f"fold_{val}"
    meta = joblib.load(fold_dir / "meta.pkl")
    sc2 = meta["scaler"]; rul_max = meta["rul_max"]
    SEQ_LEN = meta["SEQ_LEN"]
    TFT_SEEDS = meta["TFT_SEEDS"]
    BILSTM_SEEDS = meta["BILSTM_SEEDS"]
    FC_HI_fold = meta["FC_HI"]
    print(f"\n  Fold val={val}: SEQ_LEN={SEQ_LEN}, TFT={len(TFT_SEEDS)}, BiLSTM={len(BILSTM_SEEDS)}", flush=True)

    vd = df[df.bearing == val].copy()
    Xvl_full = sc2.transform(vd[FC_HI_fold].fillna(0).values)
    yvl_pw = vd["rul_pw"].values.astype(np.float32)
    yvl_raw = vd["rul_s"].values.astype(np.float32)

    vs = np.array([Xvl_full[i:i+SEQ_LEN] for i in range(len(Xvl_full)-SEQ_LEN+1)], np.float32)
    vt_pw = yvl_pw[SEQ_LEN-1:]
    vt_raw = yvl_raw[SEQ_LEN-1:]

    arch_seeds = [("tft", TFTModel, s) for s in TFT_SEEDS] + \
                 [("bilstm", BiLSTMModel, s) for s in BILSTM_SEEDS]
    fold_preds = []
    fold_archs = []
    for arch_name, cls, sd in arch_seeds:
        model = cls(len(FC_HI_fold))
        ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
        model.load_state_dict(torch.load(ckpt)); model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(vs)).numpy() * rul_max
        fold_preds.append(raw); fold_archs.append(arch_name)
    fold_preds = np.array(fold_preds)  # [n_models, n_val]

    # GPR σ
    gpr = joblib.load(fold_dir / "gpr.pkl")
    Xvl_seq_last = Xvl_full[SEQ_LEN-1:]
    gmu, gstd = gpr.predict(Xvl_seq_last, return_std=True)

    oof_preds[val] = {"preds": fold_preds, "archs": fold_archs,
                      "gstd": gstd, "yvl_raw": vt_raw,
                      "t_h": vd["t_s"].values[SEQ_LEN-1:] / 3600,
                      "hi": vd["HI"].values[SEQ_LEN-1:]}

    # 전략들
    pred_mean = fold_preds.mean(axis=0)
    pred_median = np.median(fold_preds, axis=0)
    pred_std = fold_preds.std(axis=0)
    # TFT만 / BiLSTM만 평균
    tft_idx = [i for i,a in enumerate(fold_archs) if a=="tft"]
    bilstm_idx = [i for i,a in enumerate(fold_archs) if a=="bilstm"]
    pred_tft = fold_preds[tft_idx].mean(axis=0) if tft_idx else pred_mean
    pred_bilstm = fold_preds[bilstm_idx].mean(axis=0) if bilstm_idx else pred_mean
    # weighted by individual best (학습 시점 best, 여기선 그냥 평균과 미디언)
    pred_cons = pred_median - 0.15 * gstd - 0.10 * pred_std
    pred_ultra = pred_median - 0.30 * gstd
    # Trimmed: 각 시점에서 max/min 1개 제거 → robust
    sorted_p = np.sort(fold_preds, axis=0)
    pred_trim = sorted_p[1:-1].mean(axis=0)
    # Robust median + IQR shrink (asymmetric trim)
    q25 = np.percentile(fold_preds, 25, axis=0)
    q75 = np.percentile(fold_preds, 75, axis=0)
    pred_robust = (q25 + 2*pred_median + q75) / 4

    cands = {"mean": pred_mean, "median": pred_median, "tft_only": pred_tft,
             "bilstm_only": pred_bilstm, "trim": pred_trim, "robust": pred_robust,
             "cons": pred_cons, "ultra": pred_ultra}
    cand_scores = {k: asym_score(np.clip(v, 0, None), vt_raw) for k, v in cands.items()}
    best_k = max(cand_scores, key=cand_scores.get)
    print(f"    " + "  ".join(f"{k}={v:.4f}" for k, v in cand_scores.items()), flush=True)
    print(f"    BEST: {best_k} = {cand_scores[best_k]:.4f}", flush=True)

    results.append({"val": val, **cand_scores, "best_strat": best_k,
                    "best_score": cand_scores[best_k]})

res = pd.DataFrame(results)
print("\n" + "=" * 70)
print("  v17 모델 그대로 → 폴드별 최적 전략 비교 (재학습 없음)")
print("=" * 70)
print(res.to_string(index=False))
print()
print(f"  각 전략 평균:")
for c in ["mean","median","tft_only","bilstm_only","trim","robust","cons","ultra"]:
    print(f"    {c:12s}: {res[c].mean():.4f}")
print(f"  best per-fold (oracle): {res.best_score.mean():.4f}")


# ── β 그리드서치 (asym-aware bias correction) ─────────────────────
print("\n[4] β 그리드서치 (asym-aware 보정)...", flush=True)
print("  pred *= β 의 β를 OOF에서 최적화 (각 전략별)")
betas = np.arange(0.70, 1.05, 0.01)
for strat in ["mean","median","trim","robust"]:
    scores_by_beta = []
    for beta in betas:
        all_s = []
        for val in TRAIN_NAMES:
            o = oof_preds[val]
            if strat == "mean": p = o["preds"].mean(axis=0)
            elif strat == "median": p = np.median(o["preds"], axis=0)
            elif strat == "trim":
                sp_ = np.sort(o["preds"], axis=0); p = sp_[1:-1].mean(axis=0)
            elif strat == "robust":
                q25 = np.percentile(o["preds"], 25, axis=0); q75 = np.percentile(o["preds"], 75, axis=0)
                p = (q25 + 2*np.median(o["preds"], axis=0) + q75) / 4
            p_beta = np.clip(p * beta, 0, None)
            all_s.append(asym_score(p_beta, o["yvl_raw"]))
        scores_by_beta.append(np.mean(all_s))
    best_bi = int(np.argmax(scores_by_beta))
    print(f"  {strat:8s}: β*={betas[best_bi]:.2f}  score={scores_by_beta[best_bi]:.4f}  "
          f"(β=1.00 baseline: {scores_by_beta[(np.abs(betas - 1.0)).argmin()]:.4f})")


# ── 폴드별 자동 전략 선택 후보 ─────────────────────────────────────
print("\n[5] 폴드별 HI 특성 기반 자동 선택 후보:")
for val in TRAIN_NAMES:
    o = oof_preds[val]
    hi_v = o["hi"]; n = len(hi_v)
    hi_end_slope = (hi_v[-min(10,n):].mean() - hi_v[-min(20,n):-min(10,n)].mean()) if n >= 20 else 0
    hi_mono = float(np.mean(np.diff(hi_v) > 0))
    pred_std_mean = o["preds"].std(axis=0).mean()
    pred_mean_mean = o["preds"].mean(axis=0).mean() + 1e-9
    pred_cv = pred_std_mean / pred_mean_mean
    print(f"  {val}: end_slope={hi_end_slope:.3f}  mono={hi_mono:.3f}  "
          f"pred_CV={pred_cv:.3f}  GPR_σ_mean={o['gstd'].mean():.0f}")


res.to_csv(RESULT_DIR / "lobo_v18_postprocess.csv", index=False)
print(f"\n→ {RESULT_DIR}/lobo_v18_postprocess.csv 저장")
