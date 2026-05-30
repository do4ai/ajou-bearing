"""27_EOLClassifier_LOBOCalibrated.

Train-only EOL classifier from true RUL-derived labels.

This experiment directly addresses late-prediction blowups by learning whether a
state is close to EOL, instead of relying only on RUL regression. It uses only
Train1~4 labels and LOBO validation; public Validation/Test bearings are used
for inference only.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score

RESULT_DIR = result_dir("27_EOLClassifier_LOBOCalibrated")
FEATURE_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"
ORDER_CSV = RESULT_ROOT / "14_RPMAwareOrderFeatures" / "14_rpm_order_features.csv"
ANCHOR_DEBUG = RESULT_ROOT / "05_HIBlend_Baseline_ChannelSym" / "submission_v24_v17v22_debug.csv"
SELECTOR26_DEBUG = RESULT_ROOT / "26_FinalRobust_LOBOFrozenSelector" / "26_final_robust_debug.csv"

THRESHOLDS = [1200, 2400, 3600, 6000]
PROB_CUTOFF = 0.35
TARGET_NAMES = [x.strip() for x in os.environ.get("TARGET_NAMES", "").split(",") if x.strip()] or VAL_NAMES


def load_features() -> pd.DataFrame:
    base = pd.read_csv(FEATURE_CSV).fillna(0)
    if ORDER_CSV.exists():
        order = pd.read_csv(ORDER_CSV).fillna(0)
        order_cols = [
            c for c in order.columns
            if c in {"bearing", "measurement"}
            or c.startswith("order_chsym_max_order_")
            or c.startswith("order_chsym_top2_order_")
            or c.startswith("order_chsym_range_order_")
        ]
        base = base.merge(order[order_cols], on=["bearing", "measurement"], how="left").fillna(0)
    return base


def select_features(df: pd.DataFrame) -> list[str]:
    fixed = [
        "HI", "rms_multi", "energy_ratio", "energy_std", "peak_multi", "std_multi",
        "chsym_max_kurt", "chsym_max_env_kurt", "chsym_range_kurt", "chsym_range_env_kurt",
        "HI_d1", "HI_d3", "HI_d5", "HI_slope5", "HI_slope10", "HI_acc", "HI_roll_std5",
        "rms_multi_d1", "rms_multi_d3", "rms_multi_d5", "rms_multi_slope5", "rms_multi_roll_std5",
        "energy_ratio_d1", "energy_ratio_d3", "energy_ratio_d5", "energy_ratio_slope5", "energy_ratio_roll_std5",
        "chsym_max_kurt_d1", "chsym_max_kurt_d3", "chsym_max_kurt_d5", "chsym_max_kurt_slope5",
        "chsym_max_env_kurt_d1", "chsym_max_env_kurt_d3", "chsym_max_env_kurt_d5", "chsym_max_env_kurt_slope5",
        "order_chsym_max_order_bpfo_snr", "order_chsym_max_order_bpfi_snr",
        "order_chsym_max_order_bsf_snr", "order_chsym_max_order_ftf_snr",
        "order_chsym_top2_order_bpfo_snr", "order_chsym_top2_order_bpfi_snr",
        "order_chsym_top2_order_bsf_snr", "order_chsym_top2_order_ftf_snr",
    ]
    return [c for c in fixed if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def build_models(seed: int = 27):
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed),
        "extra": ExtraTreesClassifier(n_estimators=500, max_depth=7, min_samples_leaf=2, class_weight="balanced", random_state=seed),
        "hgb": HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=12, learning_rate=0.05, l2_regularization=0.1, random_state=seed),
    }


def fit_predict(train_df: pd.DataFrame, query_df: pd.DataFrame, cols: list[str], label_col: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sc = StandardScaler().fit(train_df[cols].values)
    xtr = sc.transform(train_df[cols].values)
    xq = sc.transform(query_df[cols].values)
    y = train_df[label_col].values.astype(int)
    preds = {}
    for name, model in build_models().items():
        if len(np.unique(y)) < 2:
            preds[name] = np.zeros(len(query_df))
            continue
        model.fit(xtr, y)
        if hasattr(model, "predict_proba"):
            preds[name] = model.predict_proba(xq)[:, 1]
        else:
            preds[name] = model.predict(xq).astype(float)
    ens = np.vstack(list(preds.values())).mean(axis=0)
    return preds, ens


def score_probs(y_true: np.ndarray, prob: np.ndarray, cutoff: float) -> dict[str, float]:
    pred = prob >= cutoff
    out = {
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision_proxy": float((y_true[pred].mean() if pred.any() else 0.0)),
        "positive_rate": float(pred.mean()),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["ap"] = float(average_precision_score(y_true, prob))
    else:
        out["auc"] = np.nan
        out["ap"] = np.nan
    return out


def lobo_eval(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df.bearing.isin(TRAIN_NAMES)].copy()
    for t in THRESHOLDS:
        train[f"eol_{t}"] = (train.rul_s <= t).astype(int)
    prob_rows = []
    summary_rows = []
    for val in TRAIN_NAMES:
        tr = train[train.bearing != val].copy()
        va = train[train.bearing == val].copy()
        for t in THRESHOLDS:
            label = f"eol_{t}"
            model_probs, ens = fit_predict(tr, va, cols, label)
            y = va[label].values.astype(int)
            metrics = score_probs(y, ens, PROB_CUTOFF)
            last_prob = float(ens[-1])
            last_true = int(y[-1])
            prob_rows.append({
                "val": val,
                "threshold_s": t,
                "last_true": last_true,
                "last_prob": last_prob,
                "last_detected": int(last_prob >= PROB_CUTOFF),
                **{f"prob_{k}": float(v[-1]) for k, v in model_probs.items()},
            })
            summary_rows.append({"val": val, "threshold_s": t, **metrics})
    return pd.DataFrame(prob_rows), pd.DataFrame(summary_rows)


def public_infer(df: pd.DataFrame, cols: list[str], target_names: list[str]) -> pd.DataFrame:
    train = df[df.bearing.isin(TRAIN_NAMES)].copy()
    missing = [n for n in target_names if n not in set(df.bearing)]
    if missing:
        raise ValueError(f"TARGET_NAMES not found in feature table: {missing}. Regenerate v25/14 features for final test first.")
    test_last = df[df.bearing.isin(target_names)].groupby("bearing", sort=False).tail(1).copy()
    for t in THRESHOLDS:
        train[f"eol_{t}"] = (train.rul_s <= t).astype(int)
    rows = test_last[["bearing", "measurement", "t_s", "HI", "rms_multi", "energy_ratio"]].copy().reset_index(drop=True)
    for t in THRESHOLDS:
        _, ens = fit_predict(train, test_last, cols, f"eol_{t}")
        rows[f"p_eol_{t}"] = ens
    return rows


def integrate_submission(public_probs: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
    anchor = pd.read_csv(ANCHOR_DEBUG)[["Bearing", "RUL_blend_combined_s"]]
    anchor.columns = ["bearing", "anchor_rul_s"]
    anchor = anchor[anchor["bearing"].isin(target_names)].reset_index(drop=True)
    out = public_probs.merge(anchor, on="bearing", how="left")
    if SELECTOR26_DEBUG.exists():
        s26 = pd.read_csv(SELECTOR26_DEBUG)[["bearing", "26_final_robust_rul_s", "26_reason"]]
        s26 = s26[s26["bearing"].isin(target_names)].reset_index(drop=True)
        out = out.merge(s26, on="bearing", how="left")
    else:
        out["26_final_robust_rul_s"] = out["anchor_rul_s"]
        out["26_reason"] = "missing_26"

    preds = []
    reasons = []
    for _, r in out.iterrows():
        pred = float(r["26_final_robust_rul_s"])
        reason = str(r["26_reason"])
        # Train-only classifier rules. High recall is prioritized over precision.
        if r["p_eol_2400"] >= 0.45:
            pred = min(pred, 2400.0)
            reason += "+clf_p2400"
        elif r["p_eol_3600"] >= 0.45:
            pred = min(pred, 3600.0)
            reason += "+clf_p3600"
        elif r["p_eol_6000"] >= 0.45:
            pred = min(pred, 6000.0)
            reason += "+clf_p6000"
        preds.append(max(600.0, pred))
        reasons.append(reason)
    out["27_eolclf_rul_s"] = preds
    out["27_reason"] = reasons
    return out


def main() -> None:
    df = load_features()
    cols = select_features(df)
    lobo_probs, lobo_summary = lobo_eval(df, cols)
    lobo_probs.to_csv(RESULT_DIR / "27_eol_classifier_lobo_probs.csv", index=False)
    lobo_summary.to_csv(RESULT_DIR / "27_eol_classifier_lobo_summary.csv", index=False)

    public_probs = public_infer(df, cols, TARGET_NAMES)
    public_probs.to_csv(RESULT_DIR / "27_eol_classifier_public_probs.csv", index=False)
    integrated = integrate_submission(public_probs, TARGET_NAMES)
    integrated.to_csv(RESULT_DIR / "27_eol_classifier_integrated_debug.csv", index=False)
    sub = integrated[["bearing", "27_eolclf_rul_s", "27_reason"]].copy()
    sub.columns = ["Bearing", "RUL_pred_seconds", "Reason"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "27_eol_classifier_submission.xlsx", index=False)

    print("27_EOLClassifier_LOBOCalibrated")
    print("LOBO last detection:")
    print(lobo_probs.to_string(index=False))
    print("\nLOBO summary:")
    print(lobo_summary.groupby("threshold_s")[["recall", "positive_rate", "auc", "ap"]].mean().to_string())
    print("\nPublic integrated:")
    print(integrated[["bearing", "anchor_rul_s", "26_final_robust_rul_s", "p_eol_2400", "p_eol_3600", "p_eol_6000", "27_eolclf_rul_s", "27_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
