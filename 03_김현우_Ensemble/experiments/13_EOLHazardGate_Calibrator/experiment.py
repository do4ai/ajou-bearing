"""13_EOLHazardGate_Calibrator.

Purpose:
  Detect EOL-risk states where model predictions must not be allowed to explode.
  This script is intentionally simple and deterministic: it uses labeled Train
  feature states as a nearest-neighbor EOL risk library, then applies conservative
  clamps to existing submission candidates.

Inputs:
  results/v25_features_dynamics.csv
  results/submission_v24_v17v22_debug.csv
  results/submission_v19_blend_debug.csv
  results/submission_v8_v17v25_debug.csv
  results/submission_v9_v17v26_debug.csv

Outputs:
  results/13_eol_hazard_lobo.csv
  results/13_eol_hazard_test.csv
  results/13_eol_hazard_safe_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("13_EOLHazardGate_Calibrator")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"
DEBUG_FILES = {
    "submission_v24_v17v22_debug.csv": RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv",
    "submission_v19_blend_debug.csv": RESULT_ROOT / "03_HIBlend_Baseline_EOLDirect" / "submission_v19_blend_debug.csv",
    "submission_v8_v17v25_debug.csv": RESULT_ROOT / "08_HIBlend_Baseline_Dynamics" / "submission_v8_v17v25_debug.csv",
    "submission_v9_v17v26_debug.csv": RESULT_ROOT / "09_HIBlend_Baseline_DomainAdv" / "submission_v9_v17v26_debug.csv",
}


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / (np.sum(weights) + 1e-12)
    return float(values[np.searchsorted(cdf, quantile, side="left").clip(0, len(values) - 1)])


def feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "HI", "rms_multi", "energy_ratio", "energy_std", "peak_multi", "std_multi",
        "chsym_max_kurt", "chsym_max_env_kurt", "chsym_range_kurt", "chsym_range_env_kurt",
        "HI_d1", "HI_d3", "HI_d5", "HI_slope5", "HI_slope10", "HI_acc", "HI_roll_std5",
        "rms_multi_d1", "rms_multi_d3", "rms_multi_d5", "rms_multi_slope5", "rms_multi_roll_std5",
        "energy_ratio_d1", "energy_ratio_d3", "energy_ratio_d5", "energy_ratio_slope5", "energy_ratio_roll_std5",
        "chsym_max_kurt_d1", "chsym_max_kurt_d3", "chsym_max_kurt_d5", "chsym_max_kurt_slope5",
        "chsym_max_env_kurt_d1", "chsym_max_env_kurt_d3", "chsym_max_env_kurt_d5", "chsym_max_env_kurt_slope5",
    ]
    prefix = ("HI", "rms", "energy", "chsym", "env", "crest", "kurt", "peak", "p2p", "bpfi", "bpfo", "bsf", "ftf")
    broad = [
        c for c in df.columns
        if c not in {"bearing", "measurement", "t_s", "rul_s"}
        and pd.api.types.is_numeric_dtype(df[c])
        and (c.startswith(prefix) or any(k in c for k in ["_d", "_slope", "_acc", "_roll_std"]))
    ]
    # Keep preferred columns first for interpretability, then append all broad degradation features.
    cols = []
    for c in preferred + broad:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def knn_state(train_ref: pd.DataFrame, query: pd.DataFrame, cols: list[str], k: int = 12) -> pd.DataFrame:
    sc = StandardScaler().fit(train_ref[cols].fillna(0).values)
    x_ref = sc.transform(train_ref[cols].fillna(0).values)
    x_q = sc.transform(query[cols].fillna(0).values)
    dist = pairwise_distances(x_q, x_ref)
    rows = []
    for qi, (_, qrow) in enumerate(query.iterrows()):
        order = np.argsort(dist[qi])[:k]
        nn = train_ref.iloc[order].copy()
        d = dist[qi, order]
        w = 1.0 / (d + 1e-6)
        rul = nn["rul_s"].values.astype(float)
        p2400 = float(np.sum(w * (rul <= 2400.0)) / np.sum(w))
        p3600 = float(np.sum(w * (rul <= 3600.0)) / np.sum(w))
        p6000 = float(np.sum(w * (rul <= 6000.0)) / np.sum(w))
        q20 = weighted_quantile(rul, 0.20, w)
        q35 = weighted_quantile(rul, 0.35, w)
        q50 = weighted_quantile(rul, 0.50, w)
        rows.append({
            "bearing": qrow["bearing"],
            "measurement": int(qrow["measurement"]),
            "t_s": float(qrow.get("t_s", np.nan)),
            "HI": float(qrow.get("HI", np.nan)),
            "rms_multi": float(qrow.get("rms_multi", np.nan)),
            "energy_ratio": float(qrow.get("energy_ratio", np.nan)),
            "nn1_bearing": nn.iloc[0]["bearing"],
            "nn1_measurement": int(nn.iloc[0]["measurement"]),
            "nn1_rul_s": float(nn.iloc[0]["rul_s"]),
            "nn1_dist": float(d[0]),
            "p_eol_2400": p2400,
            "p_eol_3600": p3600,
            "p_eol_6000": p6000,
            "knn_rul_q20": q20,
            "knn_rul_q35": q35,
            "knn_rul_q50": q50,
        })
    return pd.DataFrame(rows)


def gate_prediction(anchor: float, row: pd.Series) -> tuple[float, str]:
    hi = float(row["HI"])
    energy = float(row["energy_ratio"])
    p2400 = float(row["p_eol_2400"])
    p3600 = float(row["p_eol_3600"])
    p6000 = float(row["p_eol_6000"])
    q20 = float(row["knn_rul_q20"])
    q35 = float(row["knn_rul_q35"])
    q50 = float(row["knn_rul_q50"])

    if hi >= 0.90 and (p2400 >= 0.20 or q35 <= 3600):
        return max(600.0, min(anchor, q35, 2400.0)), "high_HI_eol_clamp"
    if p2400 >= 0.35 or q20 <= 2400:
        return max(600.0, min(anchor, q35, 3600.0)), "knn_eol_clamp"
    if energy >= 20.0 and hi >= 0.35:
        return max(600.0, min(anchor, q50, 6000.0)), "high_energy_tail_clamp"
    if energy >= 15.0 and p6000 >= 0.20:
        return max(600.0, min(anchor, q50, 6000.0)), "energy_tail_clamp"
    if p3600 >= 0.45:
        return max(600.0, min(anchor, q50, 6000.0)), "moderate_eol_clamp"
    return max(600.0, anchor), "pass"


def load_debug_predictions() -> pd.DataFrame:
    frames = []
    specs = [
        ("5_HIBlend_Baseline_ChannelSym", "submission_v24_v17v22_debug.csv", "RUL_blend_combined_s"),
        ("3_HIBlend_Baseline_EOLDirect", "submission_v19_blend_debug.csv", "RUL_blend_combined_s"),
        ("8_HIBlend_Baseline_Dynamics", "submission_v8_v17v25_debug.csv", "RUL_combined_s"),
        ("9_HIBlend_Baseline_DomainAdv", "submission_v9_v17v26_debug.csv", "RUL_combined_s"),
    ]
    for method, fname, col in specs:
        p = DEBUG_FILES[fname]
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if col not in df.columns:
            continue
        frames.append(df[["Bearing", col]].rename(columns={"Bearing": "bearing", col: method}))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="bearing", how="outer")
    return out


def main() -> None:
    df = pd.read_csv(FEATURE_CSV).fillna(0)
    cols = feature_columns(df)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    test = df[df.bearing.isin(VAL_NAMES)].reset_index(drop=True)
    test_last = test.groupby("bearing", sort=False).tail(1).reset_index(drop=True)

    lobo_rows = []
    for val in TRAIN_NAMES:
        ref = train[train.bearing != val].reset_index(drop=True)
        query = train[train.bearing == val].tail(1).reset_index(drop=True)
        pred = knn_state(ref, query, cols, k=12)
        anchor = float(query.iloc[0]["rul_s"])
        # LOBO gate quality is evaluated as a stand-alone nearest-neighbor RUL estimate.
        for qname in ["knn_rul_q20", "knn_rul_q35", "knn_rul_q50"]:
            pred[f"score_{qname}"] = asym_score([pred.iloc[0][qname]], [anchor])
        pred["true_rul_s"] = anchor
        lobo_rows.append(pred)
    lobo = pd.concat(lobo_rows, ignore_index=True)
    lobo.to_csv(RESULT_DIR / "13_eol_hazard_lobo.csv", index=False)

    test_risk = knn_state(train, test_last, cols, k=12)
    debug = load_debug_predictions()
    test_risk = test_risk.merge(debug, on="bearing", how="left")
    anchor_col = "5_HIBlend_Baseline_ChannelSym"
    safe_preds = []
    reasons = []
    for _, row in test_risk.iterrows():
        anchor = float(row.get(anchor_col, row["knn_rul_q50"]))
        pred, reason = gate_prediction(anchor, row)
        safe_preds.append(pred)
        reasons.append(reason)
    test_risk["anchor_method"] = anchor_col
    test_risk["anchor_rul_s"] = test_risk[anchor_col]
    test_risk["13_EOLHazardGate_safe_rul_s"] = safe_preds
    test_risk["gate_reason"] = reasons
    test_risk.to_csv(RESULT_DIR / "13_eol_hazard_test.csv", index=False)

    sub = test_risk[["bearing", "HI", "anchor_rul_s", "13_EOLHazardGate_safe_rul_s", "gate_reason"]].copy()
    sub.columns = ["Bearing", "HI_last", "Anchor_RUL_seconds", "RUL_pred_seconds", "Gate_reason"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "13_eol_hazard_safe_submission.xlsx", index=False)

    print("13_EOLHazardGate_Calibrator")
    print(f"  features: {len(cols)}")
    print("  LOBO last-state scores:")
    print(lobo[["bearing", "true_rul_s", "knn_rul_q20", "knn_rul_q35", "knn_rul_q50", "score_knn_rul_q35"]].to_string(index=False))
    print("\n  Test gate decisions:")
    keep_cols = ["bearing", "HI", "rms_multi", "energy_ratio", "nn1_bearing", "nn1_rul_s", "p_eol_2400", "knn_rul_q35", "anchor_rul_s", "13_EOLHazardGate_safe_rul_s", "gate_reason"]
    print(test_risk[keep_cols].to_string(index=False))
    print(f"\n  Saved: {RESULT_DIR / '13_eol_hazard_test.csv'}")
    print(f"  Saved: {RESULT_DIR / '13_eol_hazard_safe_submission.xlsx'}")


if __name__ == "__main__":
    main()
