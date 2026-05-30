"""19_DomainSpecificResidual_Calibrator.

Small-data domain-specific residual heuristic. It uses source weights from 18 and
EOL risk from 13 to produce a residual-adjusted submission candidate.
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

RESULT_DIR = result_dir("19_DomainSpecificResidual_Calibrator")


def main() -> None:
    base = pd.read_csv(RESULT_ROOT / "16_ScoreAware_CalibratedEnsemble" / "16_scoreaware_debug.csv")
    weights = pd.read_csv(RESULT_ROOT / "18_MMD_CORAL_SourceWeighting" / "18_mmd_coral_source_weights.csv")
    eol_sources = {"Train1", "Train3"}
    risk = weights.assign(is_eol_source=weights.source.isin(eol_sources).astype(float)).groupby("target").apply(lambda g: float((g.weight * g.is_eol_source).sum())).reset_index(name="domain_eol_weight")
    out = base.merge(risk, left_on="bearing", right_on="target", how="left")
    preds = []
    reasons = []
    for _, r in out.iterrows():
        pred = float(r["16_balanced_rul_s"])
        if r["domain_eol_weight"] >= 0.55 and r["traj_p_eol_6000"] >= 0.15:
            pred = min(pred, float(r["traj_rul_q50"]), 8400.0)
            reasons.append("domain_eol_residual_downshift")
        else:
            reasons.append("pass")
        preds.append(max(600.0, pred))
    out["19_domain_residual_rul_s"] = preds
    out["19_reason"] = reasons
    out.to_csv(RESULT_DIR / "19_domain_residual_debug.csv", index=False)
    sub = out[["bearing", "19_domain_residual_rul_s", "19_reason"]].copy()
    sub.columns = ["Bearing", "RUL_pred_seconds", "Reason"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "19_domain_residual_submission.xlsx", index=False)
    print("19_DomainSpecificResidual_Calibrator")
    print(out[["bearing", "16_balanced_rul_s", "domain_eol_weight", "traj_p_eol_6000", "19_domain_residual_rul_s", "19_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
