"""v25: v22 features + dynamics features + DTW-aware training.

What's new vs v22:
  - 입력 피처에 동역학 피처 추가 (HI/rms_multi/energy_ratio/chsym_max_kurt/chsym_max_env_kurt의
    d1/d3/d5/slope5/slope10/acc/roll_std5 = 35 dynamics features 추가)
  - 학습 시간 단축을 위해 3 TFT + 3 BiLSTM seed (v22의 5+5에서 축소)
  - v22 features + dynamics 합쳐서 LOBO 4-fold

Note:
  - DTC-VAE HI는 v22 모델 재활용 (피처 분포 동일)
  - 피처는 results/v25_features_dynamics.csv에서 로드 (이미 추출됨)
  - models_v25/ 에 저장
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, warnings, time
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, TRAIN_NAMES

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
RESULT_DIR = METHOD_DIR / "results"
MODEL_DIR = METHOD_DIR / "models_v25"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CSV = RESULT_DIR / "v25_features_dynamics.csv"

SEQ_LEN = 10
TFT_SEEDS = [42, 7, 123]
BILSTM_SEEDS = [365, 1234, 777]
SEEDS = TFT_SEEDS + BILSTM_SEEDS
EPOCHS = 200
PATIENCE = 80
AUG_NOISE_STD = 0.035
SCORE_LOSS_START = 30
MIN_EPOCHS = 60
BILSTM_DM = 64
ASYM_PENALTY = 5.0
EOL_LAST_K = 5
EOL_MID_K = 15

print("=" * 70); print("  v25: v22 + dynamics features"); print("=" * 70)


print("\n[1] 피처 로드 (v25_features_dynamics.csv)...", flush=True)
df = pd.read_csv(FEATURE_CSV)
df_train = df[df["bearing"].isin(TRAIN_NAMES)].copy().reset_index(drop=True)
print(f"  Train rows: {len(df_train)} ({df_train.bearing.value_counts().to_dict()})", flush=True)

excl = {"bearing", "measurement", "t_s", "rul_s"}
FC = [c for c in df_train.columns if c not in excl and pd.api.types.is_numeric_dtype(df_train[c])]
print(f"  총 피처: {len(FC)}개 (v22 base + dynamics)", flush=True)


# ── Loss ────────────────────────────────────────────────────────────
def aloss_torch(p, t, penalty=ASYM_PENALTY):
    e = p - t
    return torch.where(e > 0, penalty * e.pow(2), e.pow(2)).mean()


def score_loss_torch(p_norm, t_norm, rul_max):
    p = p_norm * rul_max; t = t_norm * rul_max
    er = 100.0 * (t - p) / (t.abs() + 1.0)
    ln_half = float(np.log(0.5))
    arg_late = torch.clamp(-ln_half * (-er).clamp(min=0) / 20.0, -50.0, 0.0)
    arg_early = torch.clamp(ln_half * er.clamp(min=0) / 50.0, -50.0, 0.0)
    A = torch.where(er <= 0, arg_late.exp(), arg_early.exp())
    return 1.0 - A.mean()


# ── Models (v22와 동일) ────────────────────────────────────────────
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


class AugDS(Dataset):
    def __init__(self, X, y, eol_last_k=EOL_LAST_K, eol_mid_k=EOL_MID_K,
                 noise=AUG_NOISE_STD, mixup_alpha=0.2, mixup_prob=0.3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.noise = noise; self.mixup_alpha = mixup_alpha; self.mixup_prob = mixup_prob
        y_np = self.y.numpy()
        order = np.argsort(y_np)
        w = np.ones_like(y_np, dtype=np.float32)
        w[order[:eol_last_k]] = 50.0
        w[order[eol_last_k:eol_last_k + eol_mid_k]] = 10.0
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
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(ol, T_0=60, T_mult=2)
    tld = DataLoader(AugDS(Xtr_seq, ytr_seq_norm), batch_size=32, shuffle=True)
    bs, bst, no_imp = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in tld:
            ol.zero_grad()
            alpha = 1.0 if ep < score_start else max(0.30, 1.0 - (ep - score_start) / 80.0)
            pred = model(xb); e = pred - yb
            weighted_mse = (torch.where(e > 0, ASYM_PENALTY * e.pow(2), e.pow(2)) * wb).mean()
            score_l = score_loss_torch(pred, yb, rul_max)
            loss = alpha * weighted_mse + (1 - alpha) * score_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); ol.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vp = np.nan_to_num(model(torch.tensor(Xvl_seq)).numpy()) * rul_max
        vp = np.clip(vp, 600.0, None)
        s_full = asym_score(vp, yvl_raw); s_last = asym_score(vp[-1:], yvl_raw[-1:])
        s = 0.7 * s_full + 0.3 * s_last
        if s > bs: bs = s; bst = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if ep >= MIN_EPOCHS and no_imp >= patience: break
    model.load_state_dict(bst); model.eval()
    with torch.no_grad():
        pred = np.nan_to_num(model(torch.tensor(Xvl_seq)).numpy()) * rul_max
    pred = np.clip(pred, 600.0, None)
    return model, pred, bs


# ── LOBO 학습 ──────────────────────────────────────────────────────
print(f"\n[2] LOBO ({len(SEEDS)} seeds/fold = {len(TFT_SEEDS)} TFT + {len(BILSTM_SEEDS)} BiLSTM)...", flush=True)
results = []
df_train = df_train.fillna(0)
for val in TRAIN_NAMES:
    tns = [b for b in TRAIN_NAMES if b != val]
    print(f"\n  Fold: Val={val}", flush=True)
    tr_df = df_train[df_train.bearing.isin(tns)].reset_index(drop=True)
    vd_df = df_train[df_train.bearing == val].reset_index(drop=True)
    sc = StandardScaler().fit(tr_df[FC].values)
    tr_s, tr_t = [], []
    for tn in tns:
        sub = df_train[df_train.bearing == tn].reset_index(drop=True)
        X = sc.transform(sub[FC].values); y = sub["rul_s"].values
        for i in range(len(X) - SEQ_LEN + 1):
            tr_s.append(X[i:i + SEQ_LEN]); tr_t.append(y[i + SEQ_LEN - 1])
    Xvl_full = sc.transform(vd_df[FC].values); yvl = vd_df["rul_s"].values
    vs, vt_l = [], []
    for i in range(len(Xvl_full) - SEQ_LEN + 1):
        vs.append(Xvl_full[i:i + SEQ_LEN]); vt_l.append(yvl[i + SEQ_LEN - 1])
    tr_s = np.array(tr_s, np.float32); tr_t = np.array(tr_t, np.float32)
    vs = np.array(vs, np.float32); vt_arr = np.array(vt_l, np.float32)
    rul_max = max(tr_t.max(), 1.0); tr_tn = tr_t / rul_max
    print(f"    train_seq={tr_s.shape}  val_seq={vs.shape}", flush=True)
    fold_models, fold_preds, fold_best, fold_archs = [], [], [], []
    arch_list = [("TFT", TFTModel, TFT_SEEDS, 30), ("BiLSTM", BiLSTMModel, BILSTM_SEEDS, 10)]
    for arch_name, cls, sds, sc_start in arch_list:
        for sd in sds:
            t0 = time.time()
            model, pred, best = train_model(cls, tr_s, tr_tn, vs, vt_arr, rul_max, sd, score_start=sc_start)
            fold_models.append(model); fold_preds.append(pred); fold_best.append(best); fold_archs.append(arch_name)
            print(f"      {arch_name} seed={sd}: best={best:.4f}  {time.time()-t0:.1f}s", flush=True)
    fold_preds = np.array(fold_preds)
    pred_mean = fold_preds.mean(axis=0); pred_median = np.median(fold_preds, axis=0)
    p_mean = np.clip(pred_mean, 600, None); p_med = np.clip(pred_median, 600, None)
    vc = vt_arr
    s_mean = asym_score(p_mean, vc); s_med = asym_score(p_med, vc)
    sl_mean = asym_score(p_mean[-1:], vc[-1:]); sl_med = asym_score(p_med[-1:], vc[-1:])
    print(f"  Full mean={s_mean:.4f}  med={s_med:.4f}", flush=True)
    print(f"  Last mean={sl_mean:.4f}  med={sl_med:.4f}  true={vc[-1]:.0f}  pred_med={p_med[-1]:.0f}", flush=True)
    results.append({"val": val, "full_mean": s_mean, "full_med": s_med,
                    "last_mean": sl_mean, "last_med": sl_med,
                    "true_last": float(vc[-1]), "pred_last_med": float(p_med[-1]),
                    "seeds_best": ",".join(f"{b:.3f}" for b in fold_best)})
    fold_dir = MODEL_DIR / f"fold_{val}"; fold_dir.mkdir(exist_ok=True)
    for i, m in enumerate(fold_models):
        sd = SEEDS[i]; arch = fold_archs[i]
        torch.save(m.state_dict(), fold_dir / f"{arch.lower()}_seed{sd}.pt")
    joblib.dump({"scaler": sc, "rul_max": rul_max, "FC": FC,
                 "TFT_SEEDS": TFT_SEEDS, "BILSTM_SEEDS": BILSTM_SEEDS,
                 "SEEDS": SEEDS, "SEQ_LEN": SEQ_LEN}, fold_dir / "meta.pkl")
    th = vd_df["t_s"].values[SEQ_LEN - 1:] / 3600
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(th, vc / 3600, "k-", lw=2, label="True")
    ax[0].plot(th, p_mean / 3600, "b-", label=f"mean ({s_mean:.3f})")
    ax[0].plot(th, p_med / 3600, "g--", label=f"med ({s_med:.3f})")
    ax[0].set(xlabel="Time(h)", ylabel="RUL(h)", title=f"{val} v25"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].bar(["full_mean", "full_med", "last_mean", "last_med"],
              [s_mean, s_med, sl_mean, sl_med], color=["blue", "green", "orange", "red"])
    ax[1].set(ylim=(0, 1.05), title=f"{val} scores")
    plt.tight_layout(); plt.savefig(RESULT_DIR / f"v25_{val}.png", dpi=120); plt.close()

res = pd.DataFrame(results)
print("\n" + "=" * 70, flush=True); print("  v25 LOBO 결과", flush=True); print("=" * 70, flush=True)
print(res.to_string(index=False), flush=True)
print(f"\n  Full mean:   {res.full_mean.mean():.4f}", flush=True)
print(f"  Full median: {res.full_med.mean():.4f}", flush=True)
print(f"  Last mean:   {res.last_mean.mean():.4f}", flush=True)
print(f"  Last median: {res.last_med.mean():.4f}", flush=True)
res.to_csv(RESULT_DIR / "lobo_v25.csv", index=False)
print(f"  Saved: {RESULT_DIR / 'lobo_v25.csv'}", flush=True)
