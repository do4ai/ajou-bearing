"""v26: v25 + multi-source domain adversarial alignment (MADA-lite).

What's new vs v25:
  - Source domain = Train1~4의 각 bearing (4 source domains)
  - Domain discriminator + gradient-reversal로 베어링 비특이적(invariant) feature 학습
  - HI 기반 degradation stage(early/mid/late) 클래스 pseudo label로 stage-aware alignment 추가
  - Test(validation) 베어링은 target → grad-reverse만 적용, RUL 회귀에는 미사용
  - 학습 시간 단축: TFT만 사용 (BiLSTM 미포함), 3 seeds

Reference: Wen et al. 2021 (DANN), Tian et al. 2023 (multistage DA), Ding et al. 2022 (multi-source DA)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, warnings, time
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shared.utils import asym_score, TRAIN_NAMES, VAL_NAMES

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_DIR = Path(__file__).resolve().parent
RESULT_DIR = METHOD_DIR / "results"
MODEL_DIR = METHOD_DIR / "models_v26"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CSV = RESULT_DIR / "v25_features_dynamics.csv"

SEQ_LEN = 10
SEEDS = [42, 7, 123]
EPOCHS = 200
PATIENCE = 80
AUG_NOISE_STD = 0.035
SCORE_LOSS_START = 30
MIN_EPOCHS = 60
ASYM_PENALTY = 5.0
EOL_LAST_K = 5
EOL_MID_K = 15
LAMBDA_DA_MAX = 0.10            # domain adv 최대 가중
LAMBDA_STAGE_MAX = 0.05         # stage-aware alignment 최대 가중

print("=" * 70); print("  v26: v25 + multi-source domain adversarial (MADA-lite)"); print("=" * 70)


# ── Gradient Reversal Layer ───────────────────────────────────────
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)


def score_loss_torch(p_norm, t_norm, rul_max):
    p = p_norm * rul_max; t = t_norm * rul_max
    er = 100.0 * (t - p) / (t.abs() + 1.0)
    ln_half = float(np.log(0.5))
    arg_late = torch.clamp(-ln_half * (-er).clamp(min=0) / 20.0, -50.0, 0.0)
    arg_early = torch.clamp(ln_half * er.clamp(min=0) / 50.0, -50.0, 0.0)
    A = torch.where(er <= 0, arg_late.exp(), arg_early.exp())
    return 1.0 - A.mean()


# ── Models ─────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, ic, oc, d=1):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, 5, padding=4*d, dilation=d)
        self.bn = nn.BatchNorm1d(oc); self.act = nn.GELU(); self.drop = nn.Dropout(0.2)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        o = self.conv(x)[..., :x.shape[-1]]
        return self.drop(self.act(self.bn(o))) + self.res(x)


class TFTBackbone(nn.Module):
    """공유 feature backbone: TCN-Transformer."""
    def __init__(self, fd, dm=64, nh=4, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(fd, fd), nn.Softmax(dim=-1))
        self.tcn = nn.Sequential(*[TCNBlock(fd if i == 0 else dm, dm, d) for i, d in enumerate(dilations)])
        enc = nn.TransformerEncoderLayer(dm, nh, dim_feedforward=128, dropout=0.2,
                                          batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.feat_dim = dm
    def forward(self, x):
        w = self.vsn(x); xw = (x * w).transpose(1, 2)
        c = self.tcn(xw).transpose(1, 2); ctx = self.tr(c)[:, -1, :]
        return ctx


class V26Model(nn.Module):
    def __init__(self, fd, n_domains, n_stages=3, dm=64):
        super().__init__()
        self.backbone = TFTBackbone(fd, dm=dm)
        self.rul_head = nn.Sequential(nn.Linear(dm, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.dom_head = nn.Sequential(nn.Linear(dm, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, n_domains))
        self.stage_head = nn.Sequential(nn.Linear(dm, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, n_stages))
    def forward(self, x, lam_da=0.0, lam_stage=0.0):
        f = self.backbone(x)
        rul = self.rul_head(f).squeeze(-1)
        f_dom = grad_reverse(f, lam_da)
        dom = self.dom_head(f_dom)
        f_stg = grad_reverse(f, lam_stage)
        stg = self.stage_head(f_stg)
        return rul, dom, stg


class V26Dataset(Dataset):
    def __init__(self, X, y, dom_id, stage_id, noise=AUG_NOISE_STD,
                 mixup_alpha=0.2, mixup_prob=0.3, w=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dom = torch.tensor(dom_id, dtype=torch.long)
        self.stg = torch.tensor(stage_id, dtype=torch.long)
        self.noise = noise; self.mixup_alpha = mixup_alpha; self.mixup_prob = mixup_prob
        self.weight = torch.tensor(w, dtype=torch.float32) if w is not None else torch.ones(len(X))
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]; y = self.y[i]; w = self.weight[i]
        d = self.dom[i]; s = self.stg[i]
        # mixup은 target/dom 라벨에는 적용 안 함 (label noise 방지)
        if self.mixup_alpha > 0 and np.random.rand() < self.mixup_prob:
            j = np.random.randint(len(self))
            if self.dom[j] == d:  # same-domain mixup만
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                x = lam * x + (1 - lam) * self.X[j]
                y = lam * y + (1 - lam) * self.y[j]
                w = lam * w + (1 - lam) * self.weight[j]
        if self.noise > 0: x = x + torch.randn_like(x) * self.noise
        return x, y, w, d, s


# ── 데이터 준비 ─────────────────────────────────────────────────────
print("\n[1] 피처 로드...", flush=True)
df = pd.read_csv(FEATURE_CSV).fillna(0)
df_train = df[df["bearing"].isin(TRAIN_NAMES)].reset_index(drop=True)
df_val = df[df["bearing"].isin(VAL_NAMES)].reset_index(drop=True)
excl = {"bearing", "measurement", "t_s", "rul_s"}
FC = [c for c in df_train.columns if c not in excl and pd.api.types.is_numeric_dtype(df_train[c])]
print(f"  Train: {len(df_train)} rows, Val: {len(df_val)} rows, features: {len(FC)}", flush=True)


def assign_stage(rul_s, max_rul):
    """0=early, 1=mid, 2=late."""
    r = rul_s / max(max_rul, 1.0)
    if r > 0.66: return 0
    if r > 0.33: return 1
    return 2


# 베어링별 stage 라벨 (RUL 기준)
all_names = TRAIN_NAMES + VAL_NAMES
dom_map = {nm: i for i, nm in enumerate(all_names)}
stage_train = np.zeros(len(df_train), dtype=np.int64)
for nm in TRAIN_NAMES:
    mask = df_train.bearing == nm
    rul_max_b = df_train.loc[mask, "rul_s"].max()
    stage_train[mask.values] = [assign_stage(r, rul_max_b) for r in df_train.loc[mask, "rul_s"].values]
df_train["stage"] = stage_train
df_train["dom_id"] = [dom_map[b] for b in df_train.bearing]
# Test bearings: stage = 1 (mid) pseudo, dom_id 부여
df_val["stage"] = 1
df_val["dom_id"] = [dom_map[b] for b in df_val.bearing]


# ── LOBO 학습 ──────────────────────────────────────────────────────
print(f"\n[2] LOBO ({len(SEEDS)} TFT seeds, domain-adversarial)...", flush=True)
results = []

for val in TRAIN_NAMES:
    tns = [b for b in TRAIN_NAMES if b != val]
    print(f"\n  Fold: Val={val}", flush=True)
    tr_df = df_train[df_train.bearing.isin(tns)].reset_index(drop=True)
    vd_df = df_train[df_train.bearing == val].reset_index(drop=True)
    test_df = df_val.reset_index(drop=True)
    sc = StandardScaler().fit(tr_df[FC].values)

    # 시퀀스 구성 (베어링별)
    def build_seq(name_list, frame):
        Xs, ys, ds, ss = [], [], [], []
        for nm in name_list:
            sub = frame[frame.bearing == nm].reset_index(drop=True)
            X = sc.transform(sub[FC].values); y = sub["rul_s"].values
            d = sub["dom_id"].values; st = sub["stage"].values
            for i in range(len(X) - SEQ_LEN + 1):
                Xs.append(X[i:i + SEQ_LEN]); ys.append(y[i + SEQ_LEN - 1])
                ds.append(d[i + SEQ_LEN - 1]); ss.append(st[i + SEQ_LEN - 1])
        return (np.array(Xs, np.float32), np.array(ys, np.float32),
                np.array(ds, np.int64), np.array(ss, np.int64))

    Xtr, ytr, dtr, str_ = build_seq(tns, df_train)
    Xvl, yvl, dvl, svl = build_seq([val], df_train)
    Xte, _, dte, ste = build_seq(VAL_NAMES, df_val)
    # target: y, stage 라벨 없음 → domain adv만 사용
    rul_max = max(ytr.max(), 1.0)
    ytr_n = ytr / rul_max

    # EOL 가중치
    order = np.argsort(ytr)
    w = np.ones_like(ytr, dtype=np.float32)
    w[order[:EOL_LAST_K]] = 50.0
    w[order[EOL_LAST_K:EOL_LAST_K + EOL_MID_K]] = 10.0

    n_domains = len(all_names)
    n_stages = 3
    print(f"    train_seq={Xtr.shape}  val_seq={Xvl.shape}  test_seq={Xte.shape}", flush=True)
    print(f"    n_domains={n_domains}", flush=True)

    fold_preds = []; fold_best = []; fold_models = []
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        model = V26Model(Xtr.shape[-1], n_domains=n_domains, n_stages=n_stages)
        ol = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(ol, T_0=60, T_mult=2)
        tr_loader = DataLoader(V26Dataset(Xtr, ytr_n, dtr, str_, w=w), batch_size=32, shuffle=True)
        # Target도 같은 배치 사이즈로 순환
        te_loader = DataLoader(V26Dataset(Xte, np.zeros(len(Xte), np.float32), dte, ste),
                                batch_size=32, shuffle=True)
        te_iter = iter(te_loader)
        bs, bst, no_imp = -np.inf, None, 0
        t0 = time.time()
        for ep in range(EPOCHS):
            # lambda warm-up
            p = ep / EPOCHS
            lam_da = LAMBDA_DA_MAX * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            lam_stage = LAMBDA_STAGE_MAX * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            alpha = 1.0 if ep < SCORE_LOSS_START else max(0.30, 1.0 - (ep - SCORE_LOSS_START) / 80.0)
            model.train()
            for xb, yb, wb, db, sb in tr_loader:
                try:
                    xt, _, _, dt_b, _ = next(te_iter)
                except StopIteration:
                    te_iter = iter(te_loader)
                    xt, _, _, dt_b, _ = next(te_iter)
                if xt.shape[0] != xb.shape[0]:
                    xt = xt[:xb.shape[0]]; dt_b = dt_b[:xb.shape[0]]
                ol.zero_grad()
                pred, dom_logits, stg_logits = model(xb, lam_da=lam_da, lam_stage=lam_stage)
                e = pred - yb
                wmse = (torch.where(e > 0, ASYM_PENALTY * e.pow(2), e.pow(2)) * wb).mean()
                sl = score_loss_torch(pred, yb, rul_max)
                # source domain CE
                ce_dom_src = nn.functional.cross_entropy(dom_logits, db)
                ce_stage = nn.functional.cross_entropy(stg_logits, sb)
                # target domain
                _, dom_logits_t, _ = model(xt, lam_da=lam_da, lam_stage=lam_stage)
                ce_dom_tgt = nn.functional.cross_entropy(dom_logits_t, dt_b)
                ce_dom = 0.5 * (ce_dom_src + ce_dom_tgt)
                loss = alpha * wmse + (1 - alpha) * sl + ce_dom + ce_stage
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.); ol.step()
            sched.step(); model.eval()
            with torch.no_grad():
                vp, _, _ = model(torch.tensor(Xvl), lam_da=0.0, lam_stage=0.0)
                vp = np.nan_to_num(vp.numpy()) * rul_max
            vp = np.clip(vp, 600.0, None)
            s_full = asym_score(vp, yvl); s_last = asym_score(vp[-1:], yvl[-1:])
            s = 0.7 * s_full + 0.3 * s_last
            if s > bs: bs = s; bst = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0
            else: no_imp += 1
            if ep >= MIN_EPOCHS and no_imp >= PATIENCE: break
        model.load_state_dict(bst); model.eval()
        with torch.no_grad():
            pp, _, _ = model(torch.tensor(Xvl), lam_da=0.0, lam_stage=0.0)
            pp = np.nan_to_num(pp.numpy()) * rul_max
        pp = np.clip(pp, 600.0, None)
        fold_models.append(model); fold_preds.append(pp); fold_best.append(bs)
        print(f"      seed={sd}: best={bs:.4f}  {time.time()-t0:.1f}s", flush=True)
    fold_preds = np.array(fold_preds)
    p_mean = np.clip(fold_preds.mean(axis=0), 600, None)
    p_med = np.clip(np.median(fold_preds, axis=0), 600, None)
    s_mean = asym_score(p_mean, yvl); s_med = asym_score(p_med, yvl)
    sl_mean = asym_score(p_mean[-1:], yvl[-1:]); sl_med = asym_score(p_med[-1:], yvl[-1:])
    print(f"  Full mean={s_mean:.4f}  med={s_med:.4f}", flush=True)
    print(f"  Last mean={sl_mean:.4f}  med={sl_med:.4f}  true={yvl[-1]:.0f}  pred_med={p_med[-1]:.0f}", flush=True)
    results.append({"val": val, "full_mean": s_mean, "full_med": s_med,
                    "last_mean": sl_mean, "last_med": sl_med,
                    "true_last": float(yvl[-1]), "pred_last_med": float(p_med[-1]),
                    "seeds_best": ",".join(f"{b:.3f}" for b in fold_best)})
    fold_dir = MODEL_DIR / f"fold_{val}"; fold_dir.mkdir(exist_ok=True)
    for i, m in enumerate(fold_models):
        torch.save(m.state_dict(), fold_dir / f"v26_seed{SEEDS[i]}.pt")
    joblib.dump({"scaler": sc, "rul_max": rul_max, "FC": FC, "SEEDS": SEEDS,
                 "SEQ_LEN": SEQ_LEN, "dom_map": dom_map, "n_stages": n_stages},
                 fold_dir / "meta.pkl")

res = pd.DataFrame(results)
print("\n" + "=" * 70, flush=True); print("  v26 LOBO 결과 (domain adv)", flush=True); print("=" * 70, flush=True)
print(res.to_string(index=False), flush=True)
print(f"\n  Full mean:   {res.full_mean.mean():.4f}", flush=True)
print(f"  Full median: {res.full_med.mean():.4f}", flush=True)
print(f"  Last mean:   {res.last_mean.mean():.4f}", flush=True)
print(f"  Last median: {res.last_med.mean():.4f}", flush=True)
res.to_csv(RESULT_DIR / "lobo_v26.csv", index=False)
print(f"  Saved: {RESULT_DIR / 'lobo_v26.csv'}", flush=True)
