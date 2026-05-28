"""v18 학습 완료 후 후처리: 폴드별 전략 + β 보정 + HI-conditioned shrinkage.

학습 모델은 models_v18/fold_TrainX/ 에 있다고 가정.

핵심:
  1. 폴드별 OOF 예측 재생성
  2. asym-aware β 그리드서치 (전체 + last-only 양쪽 최적)
  3. HI-conditioned endpoint shrinkage: pred *= sigmoid(threshold - HI)
  4. 최종 전략 선택: 폴드별 LOBO에서 best, oracle/avg 비교
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
MODEL_DIR = METHOD_DIR / "models_v18"
RESULT_DIR = METHOD_DIR / "results"


# pipeline_v18.py 와 동일한 4채널 피처 추출
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

def extract(s4, rpm, torque, tf, tr):
    feats = {}
    for ci in range(4):
        feats.update(ch_feats(s4[ci], f"ch{ci}"))
    s_all = s4.astype(np.float64)
    feats["rms_multi"] = float(np.sqrt(np.mean(s_all**2)))
    feats["std_multi"] = float(np.std(s_all))
    feats["peak_multi"] = float(np.max(np.abs(s_all)))
    for i, j in [(0,1),(0,2),(0,3),(1,2)]:
        c = np.corrcoef(s4[i].astype(np.float64), s4[j].astype(np.float64))[0,1]
        feats[f"corr_{i}{j}"] = float(c) if np.isfinite(c) else 0.0
    energies = np.array([float(np.mean(s4[i].astype(np.float64)**2)) for i in range(4)])
    feats["energy_max"] = float(energies.max())
    feats["energy_min"] = float(energies.min())
    feats["energy_ratio"] = float(energies.max() / (energies.min() + 1e-10))
    feats["energy_std"] = float(energies.std())
    feats.update({"rpm": float(rpm), "torque": float(torque),
                  "tf": float(tf), "tr": float(tr),
                  "power_proxy": float(rpm * abs(torque))})
    return feats


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


def main():
    print("=" * 70); print("  v18 후처리: 전략 + β + HI-shrinkage"); print("=" * 70)

    scaler_feats = joblib.load(MODEL_DIR / "scaler_features.pkl")
    feat_meta = joblib.load(MODEL_DIR / "feature_meta.pkl")
    FC = feat_meta["FC"]; FC_HI = feat_meta["FC_HI"]
    vae = DTCVAE(len(FC), latent=4); vae.load_state_dict(torch.load(MODEL_DIR / "dtcvae.pt"))
    vae.eval()

    dfs = {}
    for nm in TRAIN_NAMES:
        sigs, op = load_bearing(nm)
        rows = []
        for i in range(len(op)):
            row = op.iloc[i]
            f = extract(sigs[i], row.rpm, row.torque, row.temp_front, row.temp_rear)
            f["t_s"] = row.t_seconds; f["rul_s"] = row.rul_seconds; f["bearing"] = nm
            rows.append(f)
        dfs[nm] = pd.DataFrame(rows); del sigs
    df = pd.concat(dfs.values(), ignore_index=True)

    Xa = scaler_feats.transform(df[FC].fillna(0).values)
    with torch.no_grad():
        _, _, _, z_all = vae.fwd(torch.tensor(Xa, dtype=torch.float32))
    z_np = z_all.numpy()
    hi = z_np[:, 0]; hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
    df["HI"] = hi
    for i in range(z_np.shape[1]): df[f"latent_{i}"] = z_np[:, i]

    # 폴드별 OOF
    oof = {}
    for val in TRAIN_NAMES:
        fold_dir = MODEL_DIR / f"fold_{val}"
        meta = joblib.load(fold_dir / "meta.pkl")
        sc2 = meta["scaler"]; rul_max = meta["rul_max"]; SEQ_LEN = meta["SEQ_LEN"]
        TFT_SEEDS = meta["TFT_SEEDS"]; BILSTM_SEEDS = meta["BILSTM_SEEDS"]
        FC_HI_f = meta["FC_HI"]

        vd = df[df.bearing == val].copy()
        Xvl = sc2.transform(vd[FC_HI_f].fillna(0).values)
        yvl = vd["rul_s"].values.astype(np.float32)
        vs = np.array([Xvl[i:i+SEQ_LEN] for i in range(len(Xvl)-SEQ_LEN+1)], np.float32)
        vt = yvl[SEQ_LEN-1:]
        hi_seq = vd["HI"].values[SEQ_LEN-1:]

        arch_seeds = [("tft", TFTModel, s) for s in TFT_SEEDS] + \
                     [("bilstm", BiLSTMModel, s) for s in BILSTM_SEEDS]
        preds = []
        for arch_name, cls, sd in arch_seeds:
            model = cls(len(FC_HI_f))
            ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
            model.load_state_dict(torch.load(ckpt)); model.eval()
            with torch.no_grad():
                raw = model(torch.tensor(vs)).numpy() * rul_max
            preds.append(np.clip(raw, 600, None))
        preds = np.array(preds)
        gpr = joblib.load(fold_dir / "gpr.pkl")
        _, gstd = gpr.predict(Xvl[SEQ_LEN-1:], return_std=True)
        oof[val] = dict(preds=preds, gstd=gstd, yvl=vt, hi=hi_seq,
                        t_h=vd["t_s"].values[SEQ_LEN-1:] / 3600)

    # ── 전략별 점수 (full + last) ─────────────────────────────────
    def strat(p, gstd):
        return dict(mean=p.mean(0), median=np.median(p,0),
                    trim=np.sort(p,0)[1:-1].mean(0),
                    cons=np.median(p,0)-0.15*gstd-0.10*p.std(0),
                    ultra=np.median(p,0)-0.30*gstd)

    print("\n  ── v18 폴드별 점수 (full / last) ──")
    summary = []
    for val in TRAIN_NAMES:
        o = oof[val]; s_ = strat(o["preds"], o["gstd"])
        s_full = {k: asym_score(np.clip(v,600,None), o["yvl"]) for k,v in s_.items()}
        s_last = {k: asym_score(np.clip(v[-1:],600,None), o["yvl"][-1:]) for k,v in s_.items()}
        print(f"  {val}:")
        for k in ["mean","median","trim","cons","ultra"]:
            print(f"    {k:8s}: full={s_full[k]:.4f}  last={s_last[k]:.4f}  pred_last={s_[k][-1]:.0f}s  true={o['yvl'][-1]:.0f}s")
        summary.append({"val": val, **{f"full_{k}":v for k,v in s_full.items()},
                        **{f"last_{k}":v for k,v in s_last.items()},
                        "true_last": float(o["yvl"][-1]), "hi_last": float(o["hi"][-1])})
    summ_df = pd.DataFrame(summary)
    print("\n  4-fold 평균:")
    for k in ["mean","median","trim","cons","ultra"]:
        print(f"    {k:8s}: full={summ_df[f'full_{k}'].mean():.4f}  last={summ_df[f'last_{k}'].mean():.4f}")

    # ── β 그리드서치 (전체 + last 가중) ─────────────────────────
    print("\n  ── β 그리드서치 (전체+last 0.5/0.5) ──")
    betas = np.arange(0.50, 1.05, 0.02)
    for strat_name in ["median", "trim", "cons"]:
        results = []
        for beta in betas:
            full_s = []; last_s = []
            for val in TRAIN_NAMES:
                o = oof[val]; s_ = strat(o["preds"], o["gstd"])
                p = np.clip(s_[strat_name] * beta, 600, None)
                full_s.append(asym_score(p, o["yvl"]))
                last_s.append(asym_score(p[-1:], o["yvl"][-1:]))
            results.append((beta, np.mean(full_s), np.mean(last_s)))
        # 최적 β (full)
        bf = max(results, key=lambda x: x[1])
        # 최적 β (last)
        bl = max(results, key=lambda x: x[2])
        # 최적 β (blended 0.5/0.5)
        bm = max(results, key=lambda x: 0.5*x[1] + 0.5*x[2])
        print(f"    {strat_name:8s}: β(full)={bf[0]:.2f} (s={bf[1]:.4f})  "
              f"β(last)={bl[0]:.2f} (s={bl[2]:.4f})  "
              f"β(blend)={bm[0]:.2f} (full={bm[1]:.4f}, last={bm[2]:.4f})")

    # ── HI-conditioned shrinkage ─────────────────────────────────
    print("\n  ── HI-conditioned shrinkage 그리드서치 ──")
    print("  pred_adj = pred * sigmoid((thr - HI) * slope) + 600 * sigmoid((HI - thr) * slope)")
    best_combo = None
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        for slope in [5, 10, 20, 40]:
            full_s = []; last_s = []
            for val in TRAIN_NAMES:
                o = oof[val]; s_ = strat(o["preds"], o["gstd"])
                base = s_["median"]; hi_v = o["hi"]
                shrink = 1.0 / (1.0 + np.exp((hi_v - thr) * slope))  # HI<thr → 1, HI>thr → 0
                p = base * shrink + 600 * (1 - shrink)
                p = np.clip(p, 600, None)
                full_s.append(asym_score(p, o["yvl"]))
                last_s.append(asym_score(p[-1:], o["yvl"][-1:]))
            full_avg = np.mean(full_s); last_avg = np.mean(last_s)
            comb = 0.5 * full_avg + 0.5 * last_avg
            if best_combo is None or comb > best_combo[0]:
                best_combo = (comb, thr, slope, full_avg, last_avg)
    print(f"    best: thr={best_combo[1]} slope={best_combo[2]}  full={best_combo[3]:.4f}  last={best_combo[4]:.4f}  combined={best_combo[0]:.4f}")

    # 최적 strategy 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, val in zip(axes.flatten(), TRAIN_NAMES):
        o = oof[val]; s_ = strat(o["preds"], o["gstd"])
        th = o["t_h"]
        base = s_["median"]; hi_v = o["hi"]
        shrink = 1.0 / (1.0 + np.exp((hi_v - best_combo[1]) * best_combo[2]))
        p_adj = np.clip(base * shrink + 600 * (1 - shrink), 600, None)
        ax.plot(th, o["yvl"]/3600, "k-", lw=2.5, label="TRUE")
        ax.plot(th, base/3600, "b-", alpha=0.7, label="median")
        ax.plot(th, p_adj/3600, "r-", lw=2, label=f"+HI-shrink")
        s_full_adj = asym_score(p_adj, o["yvl"]); s_last_adj = asym_score(p_adj[-1:], o["yvl"][-1:])
        ax.set(xlabel="Time(h)", ylabel="RUL(h)",
               title=f"{val}: full={s_full_adj:.3f}  last={s_last_adj:.3f}  pred_last={p_adj[-1]:.0f}s")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f"v18 + HI-shrink (thr={best_combo[1]}, slope={best_combo[2]})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "v18_HI_shrink.png", dpi=120); plt.close()

    summ_df.to_csv(RESULT_DIR / "v18_postprocess_summary.csv", index=False)
    print(f"\n  → {RESULT_DIR}/v18_postprocess_summary.csv")
    print(f"  → {RESULT_DIR}/v18_HI_shrink.png")


if __name__ == "__main__":
    main()
