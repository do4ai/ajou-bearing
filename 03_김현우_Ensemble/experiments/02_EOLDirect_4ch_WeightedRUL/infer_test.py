"""v18 Test 추론 → submission.xlsx (4채널 + HI-shrinkage 적용)

흐름:
  1) Test1-6 vibration.npy 로드 + 4채널 피처 추출
  2) DTC-VAE v18 로 HI/latent
  3) 4 folds × 10 seeds = 40 models 예측
  4) 폴드 간 통합 + β 보정 + HI-shrinkage
  5) 마지막 측정값을 submission

지원 옵션:
  --strategy mean|median|trim|cons     (default: median)
  --beta 0.95                          (default: 1.0)
  --hi_thr 0.85 --hi_slope 20          (default: shrinkage 비활성)
"""
import os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import load_bearing, VAL_NAMES, TRAIN_NAMES, FS, ORDERS

import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
import torch, torch.nn as nn
import joblib

METHOD_DIR = Path(__file__).resolve().parent
MODEL_DIR = METHOD_DIR / "models_v18"
RESULT_DIR = METHOD_DIR / "results"; RESULT_DIR.mkdir(exist_ok=True)

# pipeline_v18.py 동일 ----------------------------------------------
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
    rpm_v = float(rpm) if (rpm == rpm) else 800.0
    tq_v = float(torque) if (torque == torque) else -5.0
    tf_v = float(tf) if (tf == tf) else 30.0
    tr_v = float(tr) if (tr == tr) else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v)})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="median", choices=["mean","median","trim","cons"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--hi_thr", type=float, default=None, help="HI shrinkage threshold")
    ap.add_argument("--hi_slope", type=float, default=20.0)
    args = ap.parse_args()

    print("=" * 70); print(f"  v18 Test 추론 — strategy={args.strategy} β={args.beta} HI_thr={args.hi_thr}"); print("=" * 70)

    scaler_feats = joblib.load(MODEL_DIR / "scaler_features.pkl")
    feat_meta = joblib.load(MODEL_DIR / "feature_meta.pkl")
    FC = feat_meta["FC"]; FC_HI = feat_meta["FC_HI"]
    vae = DTCVAE(len(FC), latent=4); vae.load_state_dict(torch.load(MODEL_DIR / "dtcvae.pt"))
    vae.eval()

    rows_out = []
    for nm in VAL_NAMES:
        sigs, op = load_bearing(nm)
        N = len(op); print(f"\n  {nm}: {N} measurements", flush=True)
        fts = []
        for i in range(N):
            row = op.iloc[i]
            rpm = row["rpm"] if "rpm" in op.columns else np.nan
            tq = row["torque"] if "torque" in op.columns else np.nan
            tf = row["temp_front"] if "temp_front" in op.columns else np.nan
            tr = row["temp_rear"] if "temp_rear" in op.columns else np.nan
            fts.append(extract(sigs[i], rpm, tq, tf, tr))
        dft = pd.DataFrame(fts)
        for c in FC:
            if c not in dft.columns: dft[c] = 0.0
        Xa = scaler_feats.transform(dft[FC].fillna(0).values)
        with torch.no_grad():
            _, _, _, z = vae.fwd(torch.tensor(Xa, dtype=torch.float32))
        z_np = z.numpy()
        hi = z_np[:, 0]; hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
        dft["HI"] = hi
        for i in range(z_np.shape[1]): dft[f"latent_{i}"] = z_np[:, i]

        all_preds = []
        for fold in TRAIN_NAMES:
            fold_dir = MODEL_DIR / f"fold_{fold}"
            if not fold_dir.exists(): continue
            meta = joblib.load(fold_dir / "meta.pkl")
            sc2 = meta["scaler"]; rul_max = meta["rul_max"]
            SEQ_LEN = meta["SEQ_LEN"]; FC_HI_f = meta["FC_HI"]
            TFT_SEEDS = meta["TFT_SEEDS"]; BILSTM_SEEDS = meta["BILSTM_SEEDS"]
            for c in FC_HI_f:
                if c not in dft.columns: dft[c] = 0.0
            X_fold = sc2.transform(dft[FC_HI_f].fillna(0).values)
            if len(X_fold) < SEQ_LEN:
                pad = np.tile(X_fold[0:1], (SEQ_LEN - len(X_fold), 1))
                X_pad = np.vstack([pad, X_fold])
                vs = np.array([X_pad[i:i+SEQ_LEN] for i in range(len(X_pad)-SEQ_LEN+1)], np.float32)
                pred_len = len(X_fold)
            else:
                vs = np.array([X_fold[i:i+SEQ_LEN] for i in range(len(X_fold)-SEQ_LEN+1)], np.float32)
                pred_len = len(X_fold)
            arch_seeds = [("tft", TFTModel, s) for s in TFT_SEEDS] + \
                         [("bilstm", BiLSTMModel, s) for s in BILSTM_SEEDS]
            for arch_name, cls, sd in arch_seeds:
                model = cls(len(FC_HI_f))
                ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
                model.load_state_dict(torch.load(ckpt)); model.eval()
                with torch.no_grad():
                    raw = model(torch.tensor(vs)).numpy() * rul_max
                full_pred = np.zeros(pred_len, dtype=np.float32)
                offset = pred_len - len(raw)
                full_pred[:offset] = raw[0]
                full_pred[offset:] = raw
                all_preds.append(np.clip(full_pred, 600, None))
        all_preds = np.array(all_preds)

        # 전략 선택
        if args.strategy == "mean": p_base = all_preds.mean(axis=0)
        elif args.strategy == "median": p_base = np.median(all_preds, axis=0)
        elif args.strategy == "trim":
            sp_ = np.sort(all_preds, axis=0); p_base = sp_[1:-1].mean(axis=0)
        elif args.strategy == "cons":
            p_base = np.median(all_preds, axis=0) - 0.10 * all_preds.std(axis=0)

        # β 보정
        p_adj = p_base * args.beta

        # HI shrinkage
        if args.hi_thr is not None:
            shrink = 1.0 / (1.0 + np.exp((hi - args.hi_thr) * args.hi_slope))
            p_adj = p_adj * shrink + 600 * (1 - shrink)

        p_adj = np.clip(p_adj, 600, None)

        last_pred = float(p_adj[-1])
        last_base = float(p_base[-1])
        last_hi = float(hi[-1])
        rows_out.append({
            "Bearing": nm, "N_measurements": int(N),
            "Last_t_seconds": float(op["t_seconds"].iloc[-1]) if "t_seconds" in op.columns else float(N * 600),
            "RUL_pred_seconds": last_pred,
            "RUL_pred_hours": last_pred / 3600.0,
            "RUL_base_s": last_base,
            "HI_last": last_hi,
            "n_models": int(len(all_preds)),
        })
        print(f"    HI({hi[0]:.2f}→{hi[-1]:.2f})  base={last_base:.0f}s  adj={last_pred:.0f}s ({last_pred/3600:.2f}h)", flush=True)

        seq_df = pd.DataFrame({
            "measurement": np.arange(N),
            "t_seconds": op["t_seconds"].values if "t_seconds" in op.columns else np.arange(N) * 600.0,
            "HI": hi,
            "RUL_base_s": p_base,
            "RUL_adj_s": p_adj,
            "model_std_s": all_preds.std(axis=0),
        })
        seq_df.to_csv(RESULT_DIR / f"pred_{nm}_v18.csv", index=False)

    out_df = pd.DataFrame(rows_out)
    suffix = f"_{args.strategy}"
    if args.beta != 1.0: suffix += f"_b{args.beta:.2f}"
    if args.hi_thr is not None: suffix += f"_hi{args.hi_thr}"
    out_path = RESULT_DIR / f"submission_v18{suffix}.xlsx"
    out_df.to_excel(out_path, index=False)
    print(f"\n  → {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
