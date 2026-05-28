"""15_TrajectoryKNN_DTW_RUL.

Build a trajectory-similarity RUL estimate from recent feature windows.
It merges existing dynamics features with rpm-aware order features and searches
for similar Train sliding windows.

Outputs:
  results/15_trajectory_knn_lobo.csv
  results/15_trajectory_knn_test.csv
  results/15_trajectory_knn_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("15_TrajectoryKNN_DTW_RUL")
V25_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"
ORDER_CSV = RESULT_ROOT / "14_RPMAwareOrderFeatures" / "14_rpm_order_features.csv"


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cdf = np.cumsum(weights) / (np.sum(weights) + 1e-12)
    idx = np.searchsorted(cdf, quantile, side="left")
    return float(values[min(idx, len(values) - 1)])


def load_features() -> pd.DataFrame:
    base = pd.read_csv(V25_CSV).fillna(0)
    order = pd.read_csv(ORDER_CSV).fillna(0)
    order_cols = [c for c in order.columns if c.startswith("order_") or c in {"bearing", "measurement"}]
    merged = base.merge(order[order_cols], on=["bearing", "measurement"], how="left").fillna(0)
    return merged


def select_features(df: pd.DataFrame) -> list[str]:
    exact = [
        "HI", "rms_multi", "energy_ratio", "chsym_max_kurt", "chsym_max_env_kurt",
        "HI_d5", "HI_slope5", "HI_slope10", "HI_roll_std5",
        "rms_multi_d5", "rms_multi_slope5", "rms_multi_roll_std5",
        "energy_ratio_d5", "energy_ratio_slope5", "energy_ratio_roll_std5",
    ]
    order_feats = [
        c for c in df.columns
        if c.startswith("order_chsym_max_order_")
        or c.startswith("order_chsym_top2_order_")
        or c.startswith("order_chsym_range_order_")
    ]
    cols = []
    for c in exact + order_feats:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c not in cols:
            cols.append(c)
    return cols


def window_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Equal-length trajectory distance. Normalize by size so 10/20 windows are comparable.
    return float(np.sqrt(np.mean((a - b) ** 2)))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - b[j - 1]) / np.sqrt(a.shape[1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


def build_windows(frame: pd.DataFrame, names: list[str], cols: list[str], sc: StandardScaler, length: int) -> list[dict]:
    windows = []
    for name in names:
        sub = frame[frame.bearing == name].reset_index(drop=True)
        if len(sub) < length:
            continue
        x = sc.transform(sub[cols].fillna(0).values)
        for end in range(length - 1, len(sub)):
            rul = float(sub.iloc[end].get("rul_s", np.nan))
            windows.append({
                "bearing": name,
                "end_measurement": int(sub.iloc[end]["measurement"]),
                "end_t_s": float(sub.iloc[end]["t_s"]),
                "rul_s": rul,
                "x": x[end - length + 1:end + 1],
            })
    return windows


def query_window(frame: pd.DataFrame, name: str, cols: list[str], sc: StandardScaler, length: int) -> np.ndarray:
    sub = frame[frame.bearing == name].reset_index(drop=True)
    x = sc.transform(sub[cols].fillna(0).values)
    return x[-length:]


def estimate_for_query(qx: np.ndarray, windows: list[dict], top_k: int = 12, dtw_top: int = 30) -> dict:
    aligned = np.array([window_distance(qx, w["x"]) for w in windows], dtype=np.float64)
    prelim = np.argsort(aligned)[:min(dtw_top, len(windows))]
    refined = []
    for idx in prelim:
        d_dtw = dtw_distance(qx, windows[idx]["x"])
        d = 0.65 * aligned[idx] + 0.35 * d_dtw
        refined.append((idx, d, aligned[idx], d_dtw))
    refined.sort(key=lambda z: z[1])
    chosen = refined[:min(top_k, len(refined))]
    ruls = np.array([windows[idx]["rul_s"] for idx, _, _, _ in chosen], dtype=np.float64)
    dist = np.array([d for _, d, _, _ in chosen], dtype=np.float64)
    weights = 1.0 / (dist + 1e-6)
    out = {
        "nn1_bearing": windows[chosen[0][0]]["bearing"],
        "nn1_measurement": windows[chosen[0][0]]["end_measurement"],
        "nn1_rul_s": float(ruls[0]),
        "nn1_dist": float(chosen[0][1]),
        "traj_rul_q20": weighted_quantile(ruls, 0.20, weights),
        "traj_rul_q35": weighted_quantile(ruls, 0.35, weights),
        "traj_rul_q50": weighted_quantile(ruls, 0.50, weights),
        "traj_rul_q65": weighted_quantile(ruls, 0.65, weights),
        "p_eol_2400": float(np.sum(weights * (ruls <= 2400.0)) / np.sum(weights)),
        "p_eol_6000": float(np.sum(weights * (ruls <= 6000.0)) / np.sum(weights)),
    }
    for rank, (idx, d, da, dd) in enumerate(chosen[:5], start=1):
        out[f"top{rank}_bearing"] = windows[idx]["bearing"]
        out[f"top{rank}_measurement"] = windows[idx]["end_measurement"]
        out[f"top{rank}_rul_s"] = windows[idx]["rul_s"]
        out[f"top{rank}_dist"] = d
    return out


def run_length(df: pd.DataFrame, length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = select_features(df)
    train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    sc = StandardScaler().fit(train[cols].fillna(0).values)

    lobo_rows = []
    for val in TRAIN_NAMES:
        ref_names = [n for n in TRAIN_NAMES if n != val]
        windows = build_windows(train, ref_names, cols, sc, length)
        qx = query_window(train, val, cols, sc, length)
        est = estimate_for_query(qx, windows)
        true = float(train[train.bearing == val].iloc[-1]["rul_s"])
        est.update({"bearing": val, "window_len": length, "true_rul_s": true})
        for q in ["traj_rul_q20", "traj_rul_q35", "traj_rul_q50"]:
            est[f"score_{q}"] = asym_score([est[q]], [true])
        lobo_rows.append(est)

    windows_all = build_windows(train, TRAIN_NAMES, cols, sc, length)
    test_rows = []
    for name in VAL_NAMES:
        qx = query_window(df, name, cols, sc, length)
        est = estimate_for_query(qx, windows_all)
        last = df[df.bearing == name].iloc[-1]
        est.update({
            "bearing": name,
            "window_len": length,
            "HI_last": float(last["HI"]),
            "rms_multi_last": float(last["rms_multi"]),
            "energy_ratio_last": float(last["energy_ratio"]),
        })
        test_rows.append(est)
    return pd.DataFrame(lobo_rows), pd.DataFrame(test_rows)


def main() -> None:
    df = load_features()
    all_lobo = []
    all_test = []
    print("15_TrajectoryKNN_DTW_RUL")
    for length in [10, 20]:
        lobo, test = run_length(df, length)
        all_lobo.append(lobo)
        all_test.append(test)
        print(f"  window={length}: LOBO q35 mean={lobo['score_traj_rul_q35'].mean():.4f}")
    lobo_out = pd.concat(all_lobo, ignore_index=True)
    test_out = pd.concat(all_test, ignore_index=True)
    lobo_out.to_csv(RESULT_DIR / "15_trajectory_knn_lobo.csv", index=False)
    test_out.to_csv(RESULT_DIR / "15_trajectory_knn_test.csv", index=False)

    # Conservative submission candidate: use the smaller q35 across window lengths.
    sub_rows = []
    for name in VAL_NAMES:
        sub = test_out[test_out.bearing == name]
        pred = float(sub["traj_rul_q35"].min())
        best = sub.sort_values("nn1_dist").iloc[0]
        sub_rows.append({
            "Bearing": name,
            "RUL_pred_seconds": max(600.0, pred),
            "RUL_pred_hours": max(600.0, pred) / 3600.0,
            "NN1_bearing": best["nn1_bearing"],
            "NN1_rul_s": best["nn1_rul_s"],
            "P_EOL_2400_max": float(sub["p_eol_2400"].max()),
            "P_EOL_6000_max": float(sub["p_eol_6000"].max()),
        })
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_excel(RESULT_DIR / "15_trajectory_knn_submission.xlsx", index=False)

    print("  Test trajectory estimates:")
    print(test_out[["bearing", "window_len", "nn1_bearing", "nn1_rul_s", "traj_rul_q20", "traj_rul_q35", "traj_rul_q50", "p_eol_2400", "p_eol_6000"]].to_string(index=False))
    print(f"  Saved: {RESULT_DIR / '15_trajectory_knn_test.csv'}")
    print(f"  Saved: {RESULT_DIR / '15_trajectory_knn_submission.xlsx'}")


if __name__ == "__main__":
    main()
