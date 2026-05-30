"""20_UncertaintyWeighted_TargetAdaptation.

Risk-aware lower-quantile ensemble when candidate methods disagree. This is a
lightweight alternative to dynamic hybrid DA / uncertainty weighting.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()

RESULT_DIR = result_dir("20_UncertaintyWeighted_TargetAdaptation")


def main() -> None:
    df = pd.read_csv(RESULT_ROOT / "16_ScoreAware_CalibratedEnsemble" / "16_scoreaware_debug.csv")
    cand_cols = ["5_ChannelSymBlend_combined", "3_EOLDirectBlend_combined", "8_DynamicsBlend_combined", "9_DomainAdvBlend_combined", "13_EOLHazardGate_safe_rul_s", "traj_rul_q35"]
    preds = []
    rows = []
    for _, r in df.iterrows():
        vals = np.array([float(r[c]) for c in cand_cols if c in r and np.isfinite(float(r[c]))])
        spread = float(np.std(vals) / (np.mean(vals) + 1e-6))
        risk = max(float(r["traj_p_eol_2400"]), float(r["traj_p_eol_6000"]), 1.0 if r["gate_reason"] != "pass" else 0.0)
        q = 0.20 if risk >= 0.25 or spread >= 0.60 else 0.35 if risk >= 0.15 or spread >= 0.35 else 0.50
        pred = float(np.quantile(vals, q))
        pred = min(pred, float(r["16_balanced_rul_s"])) if risk >= 0.15 else pred
        preds.append(max(600.0, pred))
        rows.append({"bearing": r["bearing"], "spread": spread, "risk": risk, "quantile": q, "pred": max(600.0, pred)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "20_uncertainty_weighted_debug.csv", index=False)
    sub = out[["bearing", "pred", "quantile", "risk", "spread"]].copy()
    sub.columns = ["Bearing", "RUL_pred_seconds", "Quantile", "Risk", "Spread"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "20_uncertainty_weighted_submission.xlsx", index=False)
    print("20_UncertaintyWeighted_TargetAdaptation")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
