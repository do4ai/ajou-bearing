"""26_FinalRobust_LOBOFrozenSelector.

Final-test oriented robust selector.

Design rules:
  - No per-public-validation hand tuning.
  - All thresholds are fixed before final test labels exist.
  - Uses Train1~4 labeled data to build fallback quantile/KNN estimates.
  - Uses existing 5_HIBlend anchor if available for the target bearings.
  - Applies generic EOL risk rules, not bearing-name-specific rules.

Outputs:
  artifacts/results/26_FinalRobust_LOBOFrozenSelector/26_final_robust_lobo.csv
  artifacts/results/26_FinalRobust_LOBOFrozenSelector/26_final_robust_debug.csv
  artifacts/results/26_FinalRobust_LOBOFrozenSelector/26_final_robust_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score

RESULT_DIR = result_dir("26_FinalRobust_LOBOFrozenSelector")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"
ORDER_CSV = RESULT_ROOT / "14_RPMAwareOrderFeatures" / "14_rpm_order_features.csv"
ANCHOR_DEBUG = RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv"

SEQ_LEN = 10
KNN_K = 12
TARGET_NAMES = [x.strip() for x in os.environ.get("TARGET_NAMES", "").split(",") if x.strip()] or VAL_NAMES


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cdf = np.cumsum(weights) / (weights.sum() + 1e-12)
    idx = np.searchsorted(cdf, q, side="left")
    return float(values[min(idx, len(values) - 1)])


def load_features() -> pd.DataFrame:
    base = pd.read_csv(FEATURE_CSV).fillna(0)
    if ORDER_CSV.exists():
        order = pd.read_csv(ORDER_CSV).fillna(0)
        order_cols = [c for c in order.columns if c in {"bearing", "measurement"} or c.startswith("order_chsym_max_order_") or c.startswith("order_chsym_top2_order_")]
        base = base.merge(order[order_cols], on=["bearing", "measurement"], how="left").fillna(0)
    return base


def select_features(df: pd.DataFrame) -> list[str]:
    fixed = [
        "HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt",
        "HI_d5", "HI_slope5", "HI_slope10", "HI_roll_std5",
        "rms_multi_d5", "rms_multi_slope5", "energy_ratio_d5", "energy_ratio_slope5",
        "order_chsym_max_order_bpfo_snr", "order_chsym_max_order_bpfi_snr",
        "order_chsym_max_order_bsf_snr", "order_chsym_max_order_ftf_snr",
    ]
    return [c for c in fixed if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def make_seq_table(df: pd.DataFrame, cols: list[str], names: list[str]) -> pd.DataFrame:
    rows = []
    for b in names:
        sub = df[df.bearing == b].reset_index(drop=True)
        if len(sub) < SEQ_LEN:
            continue
        x = sub[cols].values
        for end in range(SEQ_LEN - 1, len(sub)):
            recent = x[end - SEQ_LEN + 1:end + 1]
            row = {
                "bearing": b,
                "measurement": int(sub.iloc[end].measurement),
                "t_s": float(sub.iloc[end].t_s),
            }
            if "rul_s" in sub.columns:
                row["rul_s"] = float(sub.iloc[end].rul_s)
            for i, c in enumerate(cols):
                row[f"last_{c}"] = float(recent[-1, i])
                row[f"mean_{c}"] = float(recent[:, i].mean())
                row[f"slope_{c}"] = float(recent[-1, i] - recent[0, i])
                row[f"std_{c}"] = float(recent[:, i].std())
            rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def state_knn(train: pd.DataFrame, query: pd.DataFrame, feat_cols: list[str], k: int = KNN_K) -> pd.DataFrame:
    sc = StandardScaler().fit(train[feat_cols].values)
    xtr = sc.transform(train[feat_cols].values)
    xq = sc.transform(query[feat_cols].values)
    dist = pairwise_distances(xq, xtr)
    rows = []
    for i, (_, qrow) in enumerate(query.iterrows()):
        idx = np.argsort(dist[i])[:k]
        nn = train.iloc[idx]
        d = dist[i, idx]
        w = 1 / (d + 1e-6)
        rul = nn.rul_s.values.astype(float)
        rows.append({
            "bearing": qrow.bearing,
            "knn_q20": weighted_quantile(rul, 0.20, w),
            "knn_q35": weighted_quantile(rul, 0.35, w),
            "knn_q50": weighted_quantile(rul, 0.50, w),
            "knn_p2400": float(np.sum(w * (rul <= 2400)) / w.sum()),
            "knn_p6000": float(np.sum(w * (rul <= 6000)) / w.sum()),
            "nn1_bearing": nn.iloc[0].bearing,
            "nn1_rul_s": float(nn.iloc[0].rul_s),
        })
    return pd.DataFrame(rows)


def fit_quantile(train: pd.DataFrame, feat_cols: list[str], alpha: float) -> tuple[StandardScaler, GradientBoostingRegressor]:
    sc = StandardScaler().fit(train[feat_cols].values)
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=180,
        max_depth=2,
        min_samples_leaf=4,
        random_state=int(alpha * 1000) + 26,
    )
    model.fit(sc.transform(train[feat_cols].values), train.rul_s.values)
    return sc, model


def predict_quantile(sc: StandardScaler, model: GradientBoostingRegressor, query: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    return np.clip(model.predict(sc.transform(query[feat_cols].values)), 600, None)


def generic_eol_gate(anchor: float, q20: float, q35: float, qmodel35: float, row: pd.Series) -> tuple[float, str]:
    hi = float(row.get("last_HI", row.get("HI", 0.0)))
    rms = float(row.get("last_rms_multi", row.get("rms_multi", 0.0)))
    energy = float(row.get("last_energy_ratio", row.get("energy_ratio", 0.0)))
    p2400 = float(row.get("knn_p2400", 0.0))
    p6000 = float(row.get("knn_p6000", 0.0))

    # Fixed before final labels: generic EOL criteria, no bearing-specific branch.
    if hi >= 0.90 and (p2400 >= 0.15 or q35 <= 6000):
        return max(600.0, min(anchor, q35, qmodel35, 3600.0)), "fixed_high_hi_eol"
    if energy >= 20.0 and rms >= 0.45:
        return max(600.0, min(anchor, q35, qmodel35, 6000.0)), "fixed_high_energy_eol"
    if p2400 >= 0.25:
        return max(600.0, min(anchor, q35, qmodel35, 6000.0)), "fixed_knn_eol"
    if (hi >= 0.70 or energy >= 10.0) and p6000 >= 0.35 and min(q35, qmodel35) <= 8400:
        return max(600.0, min(anchor, q35, qmodel35, 8400.0)), "fixed_tail_eol"
    return max(600.0, anchor), "pass"


def anchor_predictions(target_names: list[str]) -> pd.DataFrame | None:
    if not ANCHOR_DEBUG.exists():
        return None
    df = pd.read_csv(ANCHOR_DEBUG)
    if "RUL_blend_combined_s" not in df.columns:
        return None
    out = df[["Bearing", "RUL_blend_combined_s"]].copy()
    out.columns = ["bearing", "anchor_5_hiblend"]
    out = out[out["bearing"].isin(target_names)].reset_index(drop=True)
    return out


def lobo_eval(seq: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for val in TRAIN_NAMES:
        train = seq[(seq.bearing.isin(TRAIN_NAMES)) & (seq.bearing != val)].copy()
        query = seq[seq.bearing == val].tail(1).copy()
        sc35, m35 = fit_quantile(train, feat_cols, 0.35)
        sc20, m20 = fit_quantile(train, feat_cols, 0.20)
        p35 = float(predict_quantile(sc35, m35, query, feat_cols)[0])
        p20 = float(predict_quantile(sc20, m20, query, feat_cols)[0])
        knn = state_knn(train, query, feat_cols).iloc[0]
        # Final-safe fallback anchor for LOBO is the model median of train-only estimates.
        anchor = float(np.median([p35, knn.knn_q35, knn.knn_q50]))
        merged = query.iloc[0].copy()
        for c in knn.index:
            merged[c] = knn[c]
        pred, reason = generic_eol_gate(anchor, float(knn.knn_q20), float(knn.knn_q35), p35, merged)
        true = float(query.iloc[0].rul_s)
        rows.append({
            "bearing": val,
            "true_last": true,
            "anchor": anchor,
            "q20_model": p20,
            "q35_model": p35,
            "knn_q35": float(knn.knn_q35),
            "pred_final_safe": pred,
            "reason": reason,
            "last_score": asym_score([pred], [true]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_features()
    cols = select_features(df)
    seq_train = make_seq_table(df, cols, TRAIN_NAMES)
    feat_cols = [c for c in seq_train.columns if c.startswith(("last_", "mean_", "slope_", "std_"))]

    lobo = lobo_eval(seq_train, feat_cols)
    lobo.to_csv(RESULT_DIR / "26_final_robust_lobo.csv", index=False)

    seq_all_train = seq_train[seq_train.bearing.isin(TRAIN_NAMES)].copy()
    missing = [n for n in TARGET_NAMES if n not in set(df.bearing)]
    if missing:
        raise ValueError(f"TARGET_NAMES not found in feature table: {missing}. Regenerate v25/14 features for final test first.")
    seq_target = make_seq_table(df, cols, TARGET_NAMES).groupby("bearing", sort=False).tail(1).copy()
    sc35, m35 = fit_quantile(seq_all_train, feat_cols, 0.35)
    sc20, m20 = fit_quantile(seq_all_train, feat_cols, 0.20)
    q35_model = predict_quantile(sc35, m35, seq_target, feat_cols)
    q20_model = predict_quantile(sc20, m20, seq_target, feat_cols)
    knn = state_knn(seq_all_train, seq_target, feat_cols)
    debug = seq_target[["bearing"]].copy().reset_index(drop=True)
    debug["q20_model"] = q20_model
    debug["q35_model"] = q35_model
    debug = debug.merge(knn, on="bearing", how="left")
    anchors = anchor_predictions(TARGET_NAMES)
    if anchors is not None:
        debug = debug.merge(anchors, on="bearing", how="left")
    else:
        debug["anchor_5_hiblend"] = np.nan

    preds = []
    reasons = []
    for i, row in debug.iterrows():
        anchor = row.anchor_5_hiblend
        if not np.isfinite(anchor):
            anchor = float(np.median([row.q35_model, row.knn_q35, row.knn_q50]))
        src = seq_target[seq_target.bearing == row.bearing].iloc[0].copy()
        for c in row.index:
            src[c] = row[c]
        pred, reason = generic_eol_gate(float(anchor), float(row.knn_q20), float(row.knn_q35), float(row.q35_model), src)
        preds.append(pred)
        reasons.append(reason)
    debug["26_final_robust_rul_s"] = preds
    debug["26_reason"] = reasons
    debug.to_csv(RESULT_DIR / "26_final_robust_debug.csv", index=False)

    sub = debug[["bearing", "26_final_robust_rul_s", "26_reason"]].copy()
    sub.columns = ["Bearing", "RUL_pred_seconds", "Reason"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "26_final_robust_submission.xlsx", index=False)

    print("26_FinalRobust_LOBOFrozenSelector")
    print("LOBO last-state:")
    print(lobo.to_string(index=False))
    print("\nPublic validation run with final-safe frozen rules:")
    print(debug[["bearing", "anchor_5_hiblend", "q35_model", "knn_q35", "knn_p2400", "knn_p6000", "26_final_robust_rul_s", "26_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
