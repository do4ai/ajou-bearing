"""v17 + v18 + v20 3-way 블렌딩 + 최종 submission.

전략:
  - HI 낮음 (healthy): v17 + v20 평균 (mid-life 강한 두 모델)
  - HI 중간 (mid-life): v20 (balanced) 우세
  - HI 높음 (EOL): v18 (EOL 강) 우세

파라미터:
  thr1 (low/mid 경계): 0.3~0.6
  thr2 (mid/high 경계): 0.6~0.9
  slope: sigmoid 가파름
  beta: 글로벌 스케일
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
MODEL_DIRS = {"v17": METHOD_DIR / "models",
              "v18": METHOD_DIR / "models_v18",
              "v22": METHOD_DIR / "models_v22"}
RESULT_DIR = METHOD_DIR / "results"

# 피처 추출 (3가지)
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

def extract_v17(s4, rpm, torque, tf, tr):
    s = s4[0].astype(np.float64); s_all = s4.astype(np.float64)
    f = ch_feats(s4[0], "x")  # legacy prefix
    # Strip prefix x_ → v17 expected names
    keymap = {"x_rms":"rms","x_std":"std","x_kurt":"kurtosis","x_skew":"skewness",
              "x_peak":"peak","x_crest":"crest","x_p2p":"p2p","x_shape_f":"shape_f",
              "x_fc":"fc","x_bw":"bw","x_sk_kurt":"sk_kurt","x_env_rms":"env_rms","x_env_kurt":"env_kurt"}
    feats = {}
    for k, v in f.items():
        if k in keymap: feats[keymap[k]] = v
        elif k.startswith("x_") and ("_e" in k or "_snr" in k or "_h_ratio" in k):
            feats[k[2:]] = v  # bpfi_e 등
    feats["rms_multi"] = float(np.sqrt(np.mean(s_all**2)))
    rpm_v = float(rpm) if (rpm == rpm) else 800.0
    tq_v = float(torque) if (torque == torque) else -5.0
    tf_v = float(tf) if (tf == tf) else 30.0
    tr_v = float(tr) if (tr == tr) else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v)})
    return feats


def extract_v18(s4, rpm, torque, tf, tr):
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


def extract_v22(s4, rpm, torque, tf, tr):
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
    feats["energy_max"] = float(energies.max()); feats["energy_min"] = float(energies.min())
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


# 모델 클래스
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


# 통합 추론 함수
def get_predictions(model_dir, extract_fn, vae_cls, bearings, target_folds=None):
    """모든 베어링 × 모든 폴드 모델 → preds [bearing][fold] = [n_models, n_meas]"""
    sc = joblib.load(model_dir / "scaler_features.pkl")
    meta = joblib.load(model_dir / "feature_meta.pkl")
    FC = meta["FC"]; FC_HI = meta["FC_HI"]
    vae = vae_cls(len(FC), latent=4); vae.load_state_dict(torch.load(model_dir / "dtcvae.pt"))
    vae.eval()

    bearing_preds = {}; bearing_hi = {}
    for nm in bearings:
        sigs, op = load_bearing(nm)
        rows = []
        for i in range(len(op)):
            r = op.iloc[i]
            f = extract_fn(sigs[i], r.rpm, r.torque, r.temp_front, r.temp_rear)
            rows.append(f)
        df_b = pd.DataFrame(rows)
        for c in FC:
            if c not in df_b.columns: df_b[c] = 0.0
        Xa = sc.transform(df_b[FC].fillna(0).values)
        with torch.no_grad():
            _, _, _, z = vae.fwd(torch.tensor(Xa, dtype=torch.float32))
        hi = z[:,0].numpy(); hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
        df_b["HI"] = hi
        for i in range(z.shape[1]): df_b[f"latent_{i}"] = z[:,i].numpy()
        bearing_hi[nm] = hi

        fold_preds = {}
        folds_to_use = target_folds if target_folds else TRAIN_NAMES
        # 자기 베어링이면 자기 폴드만 (LOBO OOF). 다른 베어링이면 모든 폴드.
        if nm in TRAIN_NAMES and not target_folds:
            folds = [nm]  # OOF
        else:
            folds = folds_to_use if target_folds else TRAIN_NAMES
        for fold in folds:
            fold_dir = model_dir / f"fold_{fold}"
            if not fold_dir.exists(): continue
            fmeta = joblib.load(fold_dir / "meta.pkl")
            sc2 = fmeta["scaler"]; rul_max = fmeta["rul_max"]; SEQ_LEN = fmeta["SEQ_LEN"]
            TFT_SEEDS = fmeta["TFT_SEEDS"]; BILSTM_SEEDS = fmeta["BILSTM_SEEDS"]
            FC_HI_f = fmeta["FC_HI"]
            for c in FC_HI_f:
                if c not in df_b.columns: df_b[c] = 0.0
            X = sc2.transform(df_b[FC_HI_f].fillna(0).values)
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
                ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
                if not ckpt.exists(): continue
                model = cls(len(FC_HI_f))
                model.load_state_dict(torch.load(ckpt)); model.eval()
                with torch.no_grad():
                    raw = model(torch.tensor(vs)).numpy() * rul_max
                full = np.zeros(pred_len, dtype=np.float32)
                offset = pred_len - len(raw)
                full[:offset] = raw[0]; full[offset:] = raw
                preds.append(full)
            fold_preds[fold] = np.array(preds)
        bearing_preds[nm] = fold_preds
    return bearing_preds, bearing_hi


# 3-way 블렌딩 함수
def blend_3way(p17, p18, p22, hi, thr1, thr2, slope, beta=1.0):
    """3구간 sigmoid 블렌딩."""
    # w17 dominant in low HI: 1 → 0 as HI crosses thr1
    w17 = 1.0 / (1.0 + np.exp((hi - thr1) * slope))
    # w18 dominant in high HI: 0 → 1 as HI crosses thr2
    w18 = 1.0 / (1.0 + np.exp(-(hi - thr2) * slope))
    # w22 = 1 - w17 - w18 (clip ≥ 0)
    w22 = np.clip(1.0 - w17 - w18, 0, 1)
    # normalize
    s = w17 + w18 + w22 + 1e-12
    w17, w18, w22 = w17/s, w18/s, w22/s
    p = w17 * p17 + w18 * p18 + w22 * p22
    return np.clip(p * beta, 600, None)


def main():
    print("=" * 70); print("  v17 + v18 + v20 3-way 블렌딩"); print("=" * 70)

    # OOF 예측 (LOBO)
    print("\n[1] LOBO OOF 예측 (v17, v18, v20)...")
    preds17, hi17 = get_predictions(MODEL_DIRS["v17"], extract_v17, DTCVAE_v17, TRAIN_NAMES)
    preds18, hi18 = get_predictions(MODEL_DIRS["v18"], extract_v18, DTCVAE_v18, TRAIN_NAMES)
    preds22, hi22 = get_predictions(MODEL_DIRS["v22"], extract_v22, DTCVAE_v18, TRAIN_NAMES)

    oof = {}
    for val in TRAIN_NAMES:
        p17 = preds17[val][val]; p18 = preds18[val][val]; p22 = preds22[val][val]
        p17m = np.median(np.clip(p17, 0, None), 0)
        p18m = np.median(np.clip(p18, 600, None), 0)
        p22m = np.median(np.clip(p22, 600, None), 0)
        # HI: v18+v20 평균 (가장 풍부한 피처 기반)
        hi = 0.5 * hi18[val] + 0.5 * hi22[val]
        L = min(len(p17m), len(p18m), len(p22m), len(hi))
        _, op = load_bearing(val)
        yvl = op["rul_seconds"].values[-L:]
        oof[val] = dict(p17=p17m[-L:], p18=p18m[-L:], p22=p22m[-L:],
                        hi=hi[-L:], yvl=yvl,
                        t_h=op["t_seconds"].values[-L:]/3600)
        print(f"  {val}: L={L}  p17[-1]={p17m[-1]:.0f}  p18[-1]={p18m[-1]:.0f}  p22[-1]={p22m[-1]:.0f}",
              flush=True)

    # 그리드서치
    print("\n[2] 3-way 그리드서치...")
    records = []
    for thr1 in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for thr2 in [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            if thr2 <= thr1: continue
            for slope in [10, 20, 40, 60]:
                for beta in [0.90, 0.95, 1.00, 1.05, 1.10]:
                    full_s = []; last_s = []
                    for val in TRAIN_NAMES:
                        o = oof[val]
                        p = blend_3way(o["p17"], o["p18"], o["p22"], o["hi"],
                                        thr1, thr2, slope, beta)
                        full_s.append(asym_score(p, o["yvl"]))
                        last_s.append(asym_score(p[-1:], o["yvl"][-1:]))
                    fmean = np.mean(full_s); lmean = np.mean(last_s)
                    comb = 0.7 * fmean + 0.3 * lmean
                    records.append((thr1, thr2, slope, beta, fmean, lmean, comb))

    rec = pd.DataFrame(records, columns=["thr1","thr2","slope","beta","full","last","combined"])
    rec.to_csv(RESULT_DIR / "blend_3way_grid.csv", index=False)
    print("\n  Top 10 by full:")
    print(rec.nlargest(10, "full").to_string(index=False))
    print("\n  Top 10 by combined:")
    print(rec.nlargest(10, "combined").to_string(index=False))

    # 베이스라인
    print("\n[3] 베이스라인:")
    for name, fn in [
        ("v17 only", lambda o: np.clip(o["p17"], 600, None)),
        ("v18 only", lambda o: np.clip(o["p18"], 600, None)),
        ("v22 only", lambda o: np.clip(o["p22"], 600, None)),
        ("simple avg 3-way", lambda o: np.clip((o["p17"]+o["p18"]+o["p22"])/3, 600, None)),
    ]:
        fs = [asym_score(fn(oof[v]), oof[v]["yvl"]) for v in TRAIN_NAMES]
        ls = [asym_score(fn(oof[v])[-1:], oof[v]["yvl"][-1:]) for v in TRAIN_NAMES]
        print(f"  {name:20s}: full={np.mean(fs):.4f}  last={np.mean(ls):.4f}")

    best_full = rec.nlargest(1, "full").iloc[0]
    best_comb = rec.nlargest(1, "combined").iloc[0]

    # Test 추론
    print("\n[4] Test 추론...")
    tpreds17, thi17 = get_predictions(MODEL_DIRS["v17"], extract_v17, DTCVAE_v17, VAL_NAMES, target_folds=TRAIN_NAMES)
    tpreds18, thi18 = get_predictions(MODEL_DIRS["v18"], extract_v18, DTCVAE_v18, VAL_NAMES, target_folds=TRAIN_NAMES)
    tpreds22, thi22 = get_predictions(MODEL_DIRS["v22"], extract_v22, DTCVAE_v18, VAL_NAMES, target_folds=TRAIN_NAMES)

    rows_out = []
    for nm in VAL_NAMES:
        # 각 폴드에서 median 후 폴드 간 median
        p17m_folds = [np.median(np.clip(tpreds17[nm][f], 0, None), 0) for f in TRAIN_NAMES if f in tpreds17[nm]]
        p18m_folds = [np.median(np.clip(tpreds18[nm][f], 600, None), 0) for f in TRAIN_NAMES if f in tpreds18[nm]]
        p22m_folds = [np.median(np.clip(tpreds22[nm][f], 600, None), 0) for f in TRAIN_NAMES if f in tpreds22[nm]]
        p17m = np.median(np.array(p17m_folds), 0)
        p18m = np.median(np.array(p18m_folds), 0)
        p22m = np.median(np.array(p22m_folds), 0)
        hi = 0.5 * thi18[nm] + 0.5 * thi22[nm]
        L = min(len(p17m), len(p18m), len(p22m), len(hi))
        p17m=p17m[-L:]; p18m=p18m[-L:]; p22m=p22m[-L:]; hi=hi[-L:]
        p_full = blend_3way(p17m, p18m, p22m, hi,
                             best_full.thr1, best_full.thr2, best_full.slope, best_full.beta)
        p_comb = blend_3way(p17m, p18m, p22m, hi,
                             best_comb.thr1, best_comb.thr2, best_comb.slope, best_comb.beta)
        _, op = load_bearing(nm)
        rows_out.append({"Bearing": nm, "N_meas": len(op),
                         "Last_t_s": float(op["t_seconds"].iloc[-1]),
                         "HI_last": float(hi[-1]),
                         "RUL_v17_s": float(p17m[-1]),
                         "RUL_v18_s": float(p18m[-1]),
                         "RUL_v22_s": float(p22m[-1]),
                         "RUL_blend_full_s": float(p_full[-1]),
                         "RUL_blend_combined_s": float(p_comb[-1])})
        print(f"  {nm}: HI={hi[-1]:.2f}  v17={p17m[-1]:.0f}  v18={p18m[-1]:.0f}  v22={p22m[-1]:.0f}  "
              f"blend_full={p_full[-1]:.0f}  blend_comb={p_comb[-1]:.0f}", flush=True)

    out_df = pd.DataFrame(rows_out)
    out_full = RESULT_DIR / "submission_v23_3way_full.xlsx"
    out_comb = RESULT_DIR / "submission_v23_3way_combined.xlsx"
    fmt_full = out_df[["Bearing","N_meas","Last_t_s","HI_last","RUL_blend_full_s"]].copy()
    fmt_full["RUL_blend_full_h"] = fmt_full["RUL_blend_full_s"] / 3600
    fmt_full.columns = ["Bearing","N_measurements","Last_t_seconds","HI_last","RUL_pred_seconds","RUL_pred_hours"]
    fmt_comb = out_df[["Bearing","N_meas","Last_t_s","HI_last","RUL_blend_combined_s"]].copy()
    fmt_comb["RUL_blend_combined_h"] = fmt_comb["RUL_blend_combined_s"] / 3600
    fmt_comb.columns = ["Bearing","N_measurements","Last_t_seconds","HI_last","RUL_pred_seconds","RUL_pred_hours"]
    fmt_full.to_excel(out_full, index=False)
    fmt_comb.to_excel(out_comb, index=False)
    out_df.to_csv(RESULT_DIR / "submission_v23_debug.csv", index=False)
    print(f"\n  → {out_full}")
    print(f"  → {out_comb}")
    print(f"  best full: thr1={best_full.thr1} thr2={best_full.thr2} slope={best_full.slope} β={best_full.beta}  "
          f"full={best_full.full:.4f} last={best_full.last:.4f}")
    print(f"  best comb: thr1={best_comb.thr1} thr2={best_comb.thr2} slope={best_comb.slope} β={best_comb.beta}  "
          f"full={best_comb.full:.4f} last={best_comb.last:.4f}")


if __name__ == "__main__":
    main()
