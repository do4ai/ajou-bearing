"""v8 candidate: v17 (mid-life baseline) + v25 (dynamics-aware EOL).

v25는 v22보다 동역학 피처가 추가되어 Test5의 일시적 노이즈와 진짜 EOL을 구분 가능.
이 blend는 v5(v17+v22) 대신 사용할 수 있는 차세대 제출 후보.

Output:
  results/blend_v17v25_grid.csv
  results/blend_v17v25_LOBO.png
  results/submission_v8_v17v25_combined.xlsx
  results/submission_v8_v17v25_full.xlsx
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, TRAIN_NAMES, VAL_NAMES

import torch, torch.nn as nn
import numpy as np, pandas as pd
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
MODEL_DIR_V17 = METHOD_DIR / "models"
MODEL_DIR_V25 = METHOD_DIR / "models_v25"
RESULT_DIR = METHOD_DIR / "results"
FEATURE_CSV = RESULT_DIR / "v25_features_dynamics.csv"

print("=" * 70); print("  v8 blend: v17 (mid-life) + v25 (dynamics-aware EOL)"); print("=" * 70)


# ── 모델 클래스 (v25와 동일) ────────────────────────────────────────
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
    def __init__(self, fd, dm=64, nh=4, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(fd, fd), nn.Softmax(dim=-1))
        self.tcn = nn.Sequential(*[TCNBlock(fd if i == 0 else dm, dm, d) for i, d in enumerate(dilations)])
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


def v25_predict(df_bearing, fold_name):
    """v25 fold model로 베어링 예측."""
    fold_dir = MODEL_DIR_V25 / f"fold_{fold_name}"
    meta = joblib.load(fold_dir / "meta.pkl")
    sc = meta["scaler"]; rul_max = meta["rul_max"]; FC = meta["FC"]
    SEQ_LEN = meta["SEQ_LEN"]; SEEDS = meta["SEEDS"]
    TFT_SEEDS = meta["TFT_SEEDS"]; BILSTM_SEEDS = meta["BILSTM_SEEDS"]
    for c in FC:
        if c not in df_bearing.columns:
            df_bearing[c] = 0.0
    X = sc.transform(df_bearing[FC].fillna(0).values)
    if len(X) < SEQ_LEN:
        pad = np.tile(X[0:1], (SEQ_LEN - len(X), 1))
        X = np.vstack([pad, X])
    vs = np.array([X[i:i + SEQ_LEN] for i in range(len(X) - SEQ_LEN + 1)], np.float32)
    arch_seeds = [("tft", TFTModel, s) for s in TFT_SEEDS] + \
                 [("bilstm", BiLSTMModel, s) for s in BILSTM_SEEDS]
    preds = []
    for arch, cls, sd in arch_seeds:
        ckpt = fold_dir / f"{arch}_seed{sd}.pt"
        if not ckpt.exists(): continue
        m = cls(len(FC))
        m.load_state_dict(torch.load(ckpt)); m.eval()
        with torch.no_grad():
            preds.append(np.nan_to_num(m(torch.tensor(vs)).numpy()) * rul_max)
    return np.clip(np.median(np.array(preds), axis=0), 600, None)


# ── 피처 로드 (v25_features_dynamics에서) ───────────────────────────
print("\n[1] dynamics 피처 CSV 로드...", flush=True)
df_all = pd.read_csv(FEATURE_CSV).fillna(0)
df_train = df_all[df_all.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
df_val = df_all[df_all.bearing.isin(VAL_NAMES)].reset_index(drop=True)


# ── v17 예측 호출 (별도 blend_v17_v22.py에 있는 fold_preds_from 재사용) ─
print("\n[2] v17 예측 import...", flush=True)
sys.path.insert(0, str(METHOD_DIR))
from blend_v17_v22 import (load_all_features, fold_preds_from, MODEL_DIR_V17 as MV17,
                            extract_v17 as ex17, DTCVAE_v17, blend)
# v17은 그대로 사용. v25는 새로 호출.

print("\n[3] v17 LOBO 예측 + v25 LOBO 예측...", flush=True)
data_v17, _, _ = load_all_features(TRAIN_NAMES)
oof = {}
for val in TRAIN_NAMES:
    d = data_v17[val]
    # v17 fold 예측
    p17 = fold_preds_from(MV17, d["df17"].copy(), d["FC_HI17"], val, is_v18=False)
    p17_med = np.median(np.clip(p17, 0, None), axis=0)
    # v25 fold 예측 (dynamics CSV에서 베어링 부분 사용)
    sub = df_train[df_train.bearing == val].reset_index(drop=True)
    p25_med = v25_predict(sub.copy(), val)
    # HI: v22 기반 HI (이미 dynamics CSV에 있음)
    hi = sub["HI"].values
    yvl = sub["rul_s"].values
    L = min(len(p17_med), len(p25_med), len(hi), len(yvl))
    oof[val] = dict(p17=p17_med[-L:], p25=p25_med[-L:], hi=hi[-L:], yvl=yvl[-L:],
                    t_h=sub["t_s"].values[-L:] / 3600)
    print(f"  {val}: p17_last={p17_med[-1]:.0f}  p25_last={p25_med[-1]:.0f}  true={yvl[-1]:.0f}", flush=True)


# ── HI 그리드서치 ──────────────────────────────────────────────────
print("\n[4] HI-conditioned 블렌드 그리드서치...", flush=True)
best_full = None; best_combo_full = None
best_combined = None; best_combo_comb = None
records = []
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
    for slope in [3, 5, 10, 20, 40]:
        for beta in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
            full_s, last_s = [], []
            for val in TRAIN_NAMES:
                o = oof[val]
                p = blend(o["p17"], o["p25"], o["hi"], thr, slope, beta)
                full_s.append(asym_score(p, o["yvl"]))
                last_s.append(asym_score(p[-1:], o["yvl"][-1:]))
            fm, lm = np.mean(full_s), np.mean(last_s)
            comb = 0.7 * fm + 0.3 * lm
            records.append((thr, slope, beta, fm, lm, comb))
            if best_full is None or fm > best_full:
                best_full = fm; best_combo_full = (thr, slope, beta, fm, lm)
            if best_combined is None or comb > best_combined:
                best_combined = comb; best_combo_comb = (thr, slope, beta, fm, lm)

rec = pd.DataFrame(records, columns=["thr", "slope", "beta", "full", "last", "combined"])
rec.to_csv(RESULT_DIR / "blend_v17v25_grid.csv", index=False)
print("  Top 10 by full:")
print(rec.nlargest(10, "full").to_string(index=False))
print("\n  Top 10 by combined:")
print(rec.nlargest(10, "combined").to_string(index=False))

# baseline
v17_full = np.mean([asym_score(oof[v]["p17"], oof[v]["yvl"]) for v in TRAIN_NAMES])
v17_last = np.mean([asym_score(oof[v]["p17"][-1:], oof[v]["yvl"][-1:]) for v in TRAIN_NAMES])
v25_full = np.mean([asym_score(oof[v]["p25"], oof[v]["yvl"]) for v in TRAIN_NAMES])
v25_last = np.mean([asym_score(oof[v]["p25"][-1:], oof[v]["yvl"][-1:]) for v in TRAIN_NAMES])
print(f"\n  v17 only:  full={v17_full:.4f}  last={v17_last:.4f}")
print(f"  v25 only:  full={v25_full:.4f}  last={v25_last:.4f}")
print(f"  blend best full:      thr={best_combo_full[0]} slope={best_combo_full[1]} β={best_combo_full[2]} "
      f"full={best_combo_full[3]:.4f} last={best_combo_full[4]:.4f}")
print(f"  blend best combined:  thr={best_combo_comb[0]} slope={best_combo_comb[1]} β={best_combo_comb[2]} "
      f"full={best_combo_comb[3]:.4f} last={best_combo_comb[4]:.4f}")


# ── 시각화 + Test 추론 ─────────────────────────────────────────────
opt_thr, opt_slope, opt_beta = best_combo_comb[0], best_combo_comb[1], best_combo_comb[2]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, val in zip(axes.flatten(), TRAIN_NAMES):
    o = oof[val]
    p = blend(o["p17"], o["p25"], o["hi"], opt_thr, opt_slope, opt_beta)
    fs = asym_score(p, o["yvl"]); ls = asym_score(p[-1:], o["yvl"][-1:])
    ax.plot(o["t_h"], o["yvl"] / 3600, "k-", lw=2.5, label="TRUE")
    ax.plot(o["t_h"], o["p17"] / 3600, "b-", alpha=0.5, label="v17")
    ax.plot(o["t_h"], o["p25"] / 3600, "g-", alpha=0.5, label="v25")
    ax.plot(o["t_h"], p / 3600, "r-", lw=2, label=f"blend({fs:.3f}/{ls:.3f})")
    ax.plot(o["t_h"], o["hi"] * o["yvl"].max() / 3600, "orange", ls=":", alpha=0.5, label="HI scaled")
    ax.set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} pred={p[-1]:.0f}s true={o['yvl'][-1]:.0f}s")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle(f"v17 ⊕ v25 HI-blend (thr={opt_thr} slope={opt_slope} β={opt_beta}) "
             f"full={best_combo_comb[3]:.3f} last={best_combo_comb[4]:.3f}")
plt.tight_layout(); plt.savefig(RESULT_DIR / "blend_v17v25_LOBO.png", dpi=120); plt.close()


print("\n[5] Test 추론 (v17 + v25 ensemble)...", flush=True)
data_test_v17, _, _ = load_all_features(VAL_NAMES)
rows_out = []
for nm in VAL_NAMES:
    d17 = data_test_v17[nm]
    sub = df_val[df_val.bearing == nm].reset_index(drop=True)
    # v17 ensemble across folds
    all_p17 = []
    for fold in TRAIN_NAMES:
        p17 = fold_preds_from(MV17, d17["df17"].copy(), d17["FC_HI17"], fold, is_v18=False)
        all_p17.append(np.median(np.clip(p17, 0, None), axis=0))
    p17_ens = np.median(np.array(all_p17), axis=0)
    # v25 ensemble across folds
    all_p25 = []
    for fold in TRAIN_NAMES:
        all_p25.append(v25_predict(sub.copy(), fold))
    p25_ens = np.median(np.array(all_p25), axis=0)
    hi = sub["HI"].values
    L = min(len(p17_ens), len(p25_ens), len(hi))
    p17_ens, p25_ens, hi = p17_ens[-L:], p25_ens[-L:], hi[-L:]
    p_full = blend(p17_ens, p25_ens, hi, *best_combo_full[:3])
    p_comb = blend(p17_ens, p25_ens, hi, *best_combo_comb[:3])
    rows_out.append({"Bearing": nm, "HI_last": float(hi[-1]),
                     "RUL_v17_s": float(p17_ens[-1]),
                     "RUL_v25_s": float(p25_ens[-1]),
                     "RUL_full_s": float(p_full[-1]),
                     "RUL_combined_s": float(p_comb[-1]),
                     "RUL_combined_h": float(p_comb[-1] / 3600)})
    print(f"  {nm}: HI={hi[-1]:.2f}  v17={p17_ens[-1]:.0f}s  v25={p25_ens[-1]:.0f}s  "
          f"full={p_full[-1]:.0f}s  comb={p_comb[-1]:.0f}s", flush=True)

out_df = pd.DataFrame(rows_out)
out_df.to_csv(RESULT_DIR / "submission_v8_v17v25_debug.csv", index=False)
fmt_full = out_df[["Bearing", "HI_last", "RUL_full_s"]].copy()
fmt_full.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
fmt_comb = out_df[["Bearing", "HI_last", "RUL_combined_s"]].copy()
fmt_comb.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
fmt_full.to_excel(RESULT_DIR / "submission_v8_v17v25_full.xlsx", index=False)
fmt_comb.to_excel(RESULT_DIR / "submission_v8_v17v25_combined.xlsx", index=False)
print(f"\n  Saved blend grid: {RESULT_DIR / 'blend_v17v25_grid.csv'}")
print(f"  Saved submissions: submission_v8_v17v25_{{full,combined}}.xlsx")
print(out_df.to_string(index=False))
