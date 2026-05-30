"""23_FullConditionalMADA_Trainable.

Trainable conditional MADA-lite: small PyTorch regressor with domain adversarial
and stage heads. This is the full trainable version of experiment 17, kept small
for four-bearing data.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score

RESULT_DIR = result_dir("23_FullConditionalMADA_Trainable")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, d, ndom=10, nstage=4):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 32), nn.GELU())
        self.r = nn.Linear(32, 1)
        self.dom = nn.Linear(32, ndom)
        self.stg = nn.Linear(32, nstage)
    def forward(self, x, lam=0.0):
        z = self.f(x)
        return self.r(z).squeeze(-1), self.dom(GRL.apply(z, lam)), self.stg(GRL.apply(z, lam))


def stage(rul, maxr):
    q = rul / max(maxr, 1)
    return 3 if q <= 0.08 else 2 if q <= 0.33 else 1 if q <= 0.66 else 0


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = [c for c in df.columns if c not in {"bearing", "measurement", "t_s", "rul_s"} and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c in ["HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt", "HI_slope5", "energy_ratio_slope5"]]
    dom_map = {b: i for i, b in enumerate(TRAIN_NAMES + VAL_NAMES)}
    train = df[df.bearing.isin(TRAIN_NAMES)].copy()
    test = df[df.bearing.isin(VAL_NAMES)].copy()
    train["dom"] = train.bearing.map(dom_map)
    train["stage"] = 0
    for b in TRAIN_NAMES:
        m = train.bearing == b
        maxr = train.loc[m, "rul_s"].max()
        train.loc[m, "stage"] = [stage(r, maxr) for r in train.loc[m, "rul_s"]]
    test["dom"] = test.bearing.map(dom_map)
    test["stage"] = [3 if r.HI >= 0.9 or r.energy_ratio >= 20 else 1 for _, r in test.iterrows()]
    sc = StandardScaler().fit(train[cols].values)
    x = torch.tensor(sc.transform(train[cols].values), dtype=torch.float32)
    y_max = train.rul_s.max()
    y = torch.tensor(train.rul_s.values / y_max, dtype=torch.float32)
    dom = torch.tensor(train.dom.values, dtype=torch.long)
    stg = torch.tensor(train.stage.values, dtype=torch.long)
    xt = torch.tensor(sc.transform(test[cols].values), dtype=torch.float32)
    dt = torch.tensor(test.dom.values, dtype=torch.long)
    model = Net(len(cols), ndom=len(dom_map))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(300):
        lam = min(0.2, ep / 300 * 0.2)
        pred, dl, sl = model(x, lam)
        _, dlt, _ = model(xt, lam)
        loss = nn.functional.mse_loss(pred, y) + 0.1 * nn.functional.cross_entropy(dl, dom) + 0.1 * nn.functional.cross_entropy(sl, stg) + 0.1 * nn.functional.cross_entropy(dlt, dt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p, _, _ = model(torch.tensor(sc.transform(test.groupby("bearing", sort=False).tail(1)[cols].values), dtype=torch.float32), 0)
    pred = np.clip(p.numpy() * y_max, 600, None)
    sub = pd.DataFrame({"Bearing": VAL_NAMES, "RUL_pred_seconds": pred, "RUL_pred_hours": pred / 3600})
    sub.to_csv(RESULT_DIR / "23_conditional_mada_candidate.csv", index=False)
    sub.to_excel(RESULT_DIR / "23_conditional_mada_submission.xlsx", index=False)
    print("23_FullConditionalMADA_Trainable")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
