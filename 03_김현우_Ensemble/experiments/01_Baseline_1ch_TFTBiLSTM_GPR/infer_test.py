"""Test1-6 추론 → 챌린지 submission (v3 다중 시드 TFT 앙상블)

흐름:
  1) Test{1-6} vibration.npy + operating.csv 로드
  2) 학습 단계와 동일한 31개 피처 추출
  3) DTC-VAE로 HI/latent 추출
  4) 4 folds × 3 seeds = 12 TFT 예측의 median (outlier robust)
  5) GPR σ로 보수적 보정 (μ - 0.15·σ)
  6) 마지막 측정 시점 RUL을 챌린지 제출 (submission.xlsx)

산출물:
  results/submission.xlsx     - 챌린지 제출용
  results/pred_Test{n}.csv    - 베어링별 RUL 시퀀스
"""
import os
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
MODEL_DIR = METHOD_DIR / "models"
RESULT_DIR = METHOD_DIR / "results"; RESULT_DIR.mkdir(exist_ok=True)

# ── 피처 추출 (pipeline.py와 동일) ─────────────────────────────────
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
    rpm_v = float(rpm) if (rpm == rpm) else 800.0  # NaN 처리: 800 rpm 디폴트
    tq_v = float(torque) if (torque == torque) else -5.0
    tf_v = float(tf) if (tf == tf) else 30.0
    tr_v = float(tr) if (tr == tr) else 30.0
    feats.update({"rpm": rpm_v, "torque": tq_v, "tf": tf_v, "tr": tr_v,
                  "power_proxy": rpm_v * abs(tq_v)})
    return feats


# ── 모델 클래스 ──────────────────────────────────────────────────────
class DTCVAE(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU())
        self.mu = nn.Linear(32, latent); self.lv = nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.GELU(),
                                 nn.Linear(32, 64), nn.GELU(), nn.Linear(64, d))
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
    print("=" * 70); print("  v3 Test 추론 — 4 folds × 3 seeds = 12 TFT 앙상블"); print("=" * 70)

    scaler_feats = joblib.load(MODEL_DIR / "scaler_features.pkl")
    feat_meta = joblib.load(MODEL_DIR / "feature_meta.pkl")
    FC = feat_meta["FC"]; FC_HI = feat_meta["FC_HI"]
    vae = DTCVAE(len(FC), latent=4); vae.load_state_dict(torch.load(MODEL_DIR / "dtcvae.pt"))
    vae.eval()

    rows_out = []
    for nm in VAL_NAMES:
        sigs, op = load_bearing(nm)
        N = len(op)
        print(f"\n  {nm}: {N} measurements", flush=True)
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

        # ── 4 folds × seeds 예측 ──
        all_preds = []  # [n_models, n_measurements]
        all_gpr_sigma = []
        for fold in TRAIN_NAMES:
            fold_dir = MODEL_DIR / f"fold_{fold}"
            if not fold_dir.exists(): continue
            meta = joblib.load(fold_dir / "meta.pkl")
            sc2 = meta["scaler"]; rul_max = meta["rul_max"]
            SEQ_LEN = meta["SEQ_LEN"]; SEEDS = meta["SEEDS"]; FC_HI_fold = meta["FC_HI"]
            for c in FC_HI_fold:
                if c not in dft.columns: dft[c] = 0.0
            X_fold = sc2.transform(dft[FC_HI_fold].fillna(0).values)
            # 시퀀스 구성
            if len(X_fold) < SEQ_LEN:
                # 부족하면 처음 측정으로 패딩
                pad = np.tile(X_fold[0:1], (SEQ_LEN - len(X_fold), 1))
                X_pad = np.vstack([pad, X_fold])
                vs = np.array([X_pad[i:i+SEQ_LEN] for i in range(len(X_pad)-SEQ_LEN+1)], np.float32)
                pred_len = len(X_fold)
            else:
                vs = np.array([X_fold[i:i+SEQ_LEN] for i in range(len(X_fold)-SEQ_LEN+1)], np.float32)
                pred_len = len(X_fold)
            # v11: TFT + BiLSTM 둘 다 로드
            arch_seeds = []
            if "TFT_SEEDS" in meta:
                arch_seeds.extend([("tft", TFTModel, s) for s in meta["TFT_SEEDS"]])
                arch_seeds.extend([("bilstm", BiLSTMModel, s) for s in meta["BILSTM_SEEDS"]])
            else:
                arch_seeds.extend([("tft", TFTModel, s) for s in SEEDS])
            fold_seed_preds = []
            for arch_name, cls, sd in arch_seeds:
                model = cls(len(FC_HI_fold))
                ckpt = fold_dir / f"{arch_name}_seed{sd}.pt"
                if not ckpt.exists():
                    # 호환성: 옛 tft_seedX.pt 이름
                    ckpt = fold_dir / f"tft_seed{sd}.pt"
                model.load_state_dict(torch.load(ckpt))
                model.eval()
                with torch.no_grad():
                    raw = model(torch.tensor(vs)).numpy() * rul_max
                full_pred = np.zeros(pred_len, dtype=np.float32)
                offset = pred_len - len(raw)
                full_pred[:offset] = raw[0]
                full_pred[offset:] = raw
                fold_seed_preds.append(full_pred)
                all_preds.append(full_pred)
            # GPR σ (보수적 보정용)
            gpr = joblib.load(fold_dir / "gpr.pkl")
            gmu, gstd = gpr.predict(X_fold, return_std=True)
            all_gpr_sigma.append(gstd)

        all_preds = np.array(all_preds)              # [n_models, N]
        all_gpr_sigma = np.array(all_gpr_sigma).mean(axis=0)  # [N]

        # 통계: mean, median
        pred_mean = all_preds.mean(axis=0)
        pred_median = np.median(all_preds, axis=0)
        pred_std = all_preds.std(axis=0)
        # 보수적: median - 0.15·σ_GPR - 0.10·σ_models (불확실성 두 종류)
        pred_cons = pred_median - 0.15 * all_gpr_sigma - 0.10 * pred_std
        pred_cons = np.clip(pred_cons, 0, None)
        pred_mean = np.clip(pred_mean, 0, None)
        pred_median = np.clip(pred_median, 0, None)

        last_med = float(pred_median[-1])
        last_cons = float(pred_cons[-1])
        last_mean = float(pred_mean[-1])
        rows_out.append({
            "Bearing": nm,
            "N_measurements": int(N),
            "Last_t_seconds": float(op["t_seconds"].iloc[-1]) if "t_seconds" in op.columns else float(N * 600),
            "RUL_pred_seconds": last_med,         # 권장 제출값
            "RUL_pred_hours": last_med / 3600.0,
            "RUL_mean_s": last_mean,
            "RUL_cons_s": last_cons,
            "HI_last": float(hi[-1]),
            "n_models": int(len(all_preds)),
        })
        print(f"    HI({hi[0]:.2f}→{hi[-1]:.2f})  Pred RUL median={last_med:.0f}s "
              f"({last_med/3600:.2f}h)  cons={last_cons:.0f}s  mean={last_mean:.0f}s", flush=True)

        seq_df = pd.DataFrame({
            "measurement": np.arange(N),
            "t_seconds": op["t_seconds"].values if "t_seconds" in op.columns else np.arange(N) * 600.0,
            "HI": hi,
            "RUL_median_s": pred_median,
            "RUL_mean_s": pred_mean,
            "RUL_cons_s": pred_cons,
            "model_std_s": pred_std,
            "gpr_sigma_s": all_gpr_sigma,
        })
        seq_df.to_csv(RESULT_DIR / f"pred_{nm}.csv", index=False)

    out_df = pd.DataFrame(rows_out)
    out_path = RESULT_DIR / "submission.xlsx"
    out_df.to_excel(out_path, index=False)
    print(f"\n  → {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
