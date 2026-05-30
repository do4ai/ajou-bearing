"""28_EOLRegressor_Specialist.

목적:
  EOL 영역(rul_s <= 15000s) 데이터만으로 train-based RUL 회귀기 학습.
  임의 clamp (2400/3600/6000) 대체용 정직한 회귀값 생성.

핵심 원칙:
  - 모든 출력은 Train data로 학습된 모델의 회귀 결과
  - 임의 숫자 박기 금지
  - 600s 물리 하한 클립만 허용

알고리즘:
  1. Train1~4의 rul_s <= 15000 데이터 필터링 (EOL 구간)
  2. Input features:
     - v25_features_dynamics.csv (271 cols: HI/dynamics/chsym/etc.)
     - 14_rpm_order_features.csv (rpm-aware order energies)
  3. 모델 앙상블:
     - GradientBoostingRegressor (quantile loss q=0.4) — 비대칭 페널티 직접 반영
     - RandomForestRegressor (n_estimators=500, max_depth=8)
     - ExtraTreesRegressor (variance reduction)
  4. LOBO 4-fold 검증: train의 한 베어링을 빼고 학습 → 빠진 베어링 EOL last asym_score
  5. 최종 모델: 4 베어링 전체로 재학습 → Test inference

Outputs:
  artifacts/models/28_EOLRegressor_Specialist/{gbm_q4,gbm_q5,rf,et}.pkl
  artifacts/results/28_EOLRegressor_Specialist/28_eol_regressor_lobo.csv
  artifacts/results/28_EOLRegressor_Specialist/28_eol_regressor_test.csv
  artifacts/results/28_EOLRegressor_Specialist/28_eol_regressor_submission.xlsx
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paths import RESULT_ROOT, MODEL_ROOT, add_repo_to_path, result_dir
add_repo_to_path()
from shared.utils import TRAIN_NAMES, VAL_NAMES, asym_score  # noqa: E402

RESULT_DIR = result_dir("28_EOLRegressor_Specialist")
MODEL_DIR = MODEL_ROOT / "28_EOLRegressor_Specialist"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

V25_CSV = RESULT_ROOT / "06_Dynamics_DTW_TFTBiLSTM" / "v25_features_dynamics.csv"
ORDER_CSV = RESULT_ROOT / "14_RPMAwareOrderFeatures" / "14_rpm_order_features.csv"

EOL_THRESHOLD = 15000.0          # rul_s <= 15000s 만 학습 (EOL 구간 정의)
LAST_K_FOR_SCORE = 5             # LOBO 평가용 마지막 K개 측정 평균
SEED = 42


def load_merged_features() -> pd.DataFrame:
    """v25 dynamics + 14 rpm-aware order features merge."""
    base = pd.read_csv(V25_CSV).fillna(0)
    if ORDER_CSV.exists():
        order = pd.read_csv(ORDER_CSV).fillna(0)
        order_cols = [c for c in order.columns if c.startswith("order_") or c in {"bearing", "measurement"}]
        merged = base.merge(order[order_cols], on=["bearing", "measurement"], how="left").fillna(0)
    else:
        merged = base
    return merged


def select_features(df: pd.DataFrame) -> list[str]:
    """EOL specialist 학습용 feature 선정."""
    exclude = {"bearing", "measurement", "t_s", "rul_s", "rpm_used", "torque",
               "temp_front", "temp_rear", "rpm", "tf", "tr"}
    cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def train_eol_models(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """EOL 전용 회귀 모델 4종 학습.

    quantile q=0.4 → 비대칭 페널티상 보수 측 (짧게 예측 → 늦은 예측 회피)
    quantile q=0.5 → 중앙값 (보조)
    """
    models = {}

    gbm_q4 = GradientBoostingRegressor(
        loss="quantile", alpha=0.4,
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=3, random_state=seed,
    )
    gbm_q4.fit(X, y)
    models["gbm_q4"] = gbm_q4

    gbm_q5 = GradientBoostingRegressor(
        loss="quantile", alpha=0.5,
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=3, random_state=seed,
    )
    gbm_q5.fit(X, y)
    models["gbm_q5"] = gbm_q5

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=2,
        n_jobs=-1, random_state=seed,
    )
    rf.fit(X, y)
    models["rf"] = rf

    et = ExtraTreesRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=2,
        n_jobs=-1, random_state=seed,
    )
    et.fit(X, y)
    models["et"] = et

    return models


def predict_ensemble(models: dict, X: np.ndarray) -> dict:
    """앙상블 예측: 각 모델 + median."""
    preds = {name: m.predict(X) for name, m in models.items()}
    arr = np.stack([preds["gbm_q4"], preds["gbm_q5"], preds["rf"], preds["et"]], axis=0)
    preds["ensemble_median"] = np.median(arr, axis=0)
    preds["ensemble_mean"] = arr.mean(axis=0)
    # 보수 측 (quantile q4 가중): 비대칭 페널티 직접 반영
    preds["ensemble_conservative"] = 0.5 * preds["gbm_q4"] + 0.3 * preds["rf"] + 0.2 * preds["et"]
    return preds


def main() -> None:
    print("=" * 70)
    print("28_EOLRegressor_Specialist")
    print("=" * 70)

    df = load_merged_features()
    feat_cols = select_features(df)
    print(f"  features: {len(feat_cols)}")

    df_train = df[df.bearing.isin(TRAIN_NAMES)].reset_index(drop=True)
    df_test = df[df.bearing.isin(VAL_NAMES)].reset_index(drop=True)

    # EOL filter: rul_s <= 15000s
    eol_mask = df_train["rul_s"] <= EOL_THRESHOLD
    df_eol = df_train[eol_mask].reset_index(drop=True)
    print(f"  Train EOL samples (rul_s ≤ {EOL_THRESHOLD:.0f}s): {len(df_eol)} / {len(df_train)}")
    for nm in TRAIN_NAMES:
        n = int((df_eol.bearing == nm).sum())
        print(f"    {nm}: {n} EOL measurements")

    # ── LOBO 4-fold 검증 ────────────────────────────────────────
    print("\n[LOBO 4-fold validation]")
    lobo_rows = []
    oof_preds = {}  # bearing -> ensemble_conservative pred for each EOL measurement
    for val in TRAIN_NAMES:
        eol_tr = df_eol[df_eol.bearing != val].reset_index(drop=True)
        eol_vl = df_eol[df_eol.bearing == val].reset_index(drop=True)
        if len(eol_vl) == 0:
            print(f"  Fold {val}: 0 EOL samples, skip")
            continue

        sc = StandardScaler().fit(eol_tr[feat_cols].fillna(0).values)
        Xtr = sc.transform(eol_tr[feat_cols].fillna(0).values)
        ytr = eol_tr["rul_s"].values
        Xvl = sc.transform(eol_vl[feat_cols].fillna(0).values)
        yvl = eol_vl["rul_s"].values

        models = train_eol_models(Xtr, ytr, seed=SEED)
        preds = predict_ensemble(models, Xvl)

        # 600s 물리 하한 클립 (측정 간격)
        for k in preds:
            preds[k] = np.clip(preds[k], 600.0, None)

        # full LOBO (모든 EOL 측정)
        scores = {}
        for k, p in preds.items():
            scores[f"full_{k}"] = float(asym_score(p, yvl))
        # last LOBO (마지막 5측정 평균)
        last_n = min(LAST_K_FOR_SCORE, len(yvl))
        for k, p in preds.items():
            scores[f"last_{k}"] = float(asym_score(p[-last_n:], yvl[-last_n:]))
        # 폭주 카운트 (pred > 5000s when true < 2000s)
        for k, p in preds.items():
            mask = yvl < 2000.0
            explode = int(np.sum((p[mask] > 5000.0)))
            scores[f"explode_{k}"] = explode

        row = {"val": val, "n_eol": len(eol_vl), "true_last_5_mean": float(yvl[-last_n:].mean())}
        row.update(scores)
        # Also save the conservative ensemble prediction for the last measurement
        row["pred_last_conservative"] = float(preds["ensemble_conservative"][-1])
        row["pred_last_median"] = float(preds["ensemble_median"][-1])
        lobo_rows.append(row)

        oof_preds[val] = {
            "true": yvl, "preds": preds, "rul_max": ytr.max(),
            "measurements": eol_vl["measurement"].values,
            "t_s": eol_vl["t_s"].values,
        }

        print(f"  Fold {val}: full_cons={scores['full_ensemble_conservative']:.4f}  "
              f"last_cons={scores['last_ensemble_conservative']:.4f}  "
              f"explode_cons={scores['explode_ensemble_conservative']}  "
              f"pred_last={row['pred_last_conservative']:.0f}  true_last≈{row['true_last_5_mean']:.0f}")

    lobo_df = pd.DataFrame(lobo_rows)
    lobo_df.to_csv(RESULT_DIR / "28_eol_regressor_lobo.csv", index=False)

    # 평균 출력
    print("\n  LOBO averages (across folds):")
    for k in ["full_ensemble_conservative", "last_ensemble_conservative",
              "full_ensemble_median", "last_ensemble_median",
              "full_gbm_q4", "last_gbm_q4", "full_rf", "last_rf"]:
        if k in lobo_df.columns:
            print(f"    {k}: {lobo_df[k].mean():.4f}")

    # ── 최종 모델 학습 (4 베어링 전체 EOL 데이터) ────────────────
    print("\n[Final model: train on all 4 bearings' EOL data]")
    sc_final = StandardScaler().fit(df_eol[feat_cols].fillna(0).values)
    Xall = sc_final.transform(df_eol[feat_cols].fillna(0).values)
    yall = df_eol["rul_s"].values
    final_models = train_eol_models(Xall, yall, seed=SEED)

    joblib.dump(sc_final, MODEL_DIR / "scaler.pkl")
    joblib.dump({"feat_cols": feat_cols, "EOL_THRESHOLD": EOL_THRESHOLD, "seed": SEED},
                MODEL_DIR / "meta.pkl")
    for name, m in final_models.items():
        joblib.dump(m, MODEL_DIR / f"{name}.pkl")

    # ── Test inference ────────────────────────────────────────────
    print("\n[Test inference]")
    test_last = df_test.groupby("bearing", sort=False).tail(1).reset_index(drop=True)
    Xte = sc_final.transform(test_last[feat_cols].fillna(0).values)
    test_preds = predict_ensemble(final_models, Xte)
    # 600s 물리 하한 클립
    for k in test_preds:
        test_preds[k] = np.clip(test_preds[k], 600.0, None)

    rows_out = []
    for i, (_, r) in enumerate(test_last.iterrows()):
        row = {
            "Bearing": r["bearing"],
            "HI_last": float(r["HI"]),
            "rms_multi_last": float(r["rms_multi"]),
            "energy_ratio_last": float(r["energy_ratio"]),
            "EOL_gbm_q4_s": float(test_preds["gbm_q4"][i]),
            "EOL_gbm_q5_s": float(test_preds["gbm_q5"][i]),
            "EOL_rf_s": float(test_preds["rf"][i]),
            "EOL_et_s": float(test_preds["et"][i]),
            "EOL_median_s": float(test_preds["ensemble_median"][i]),
            "EOL_mean_s": float(test_preds["ensemble_mean"][i]),
            "EOL_conservative_s": float(test_preds["ensemble_conservative"][i]),
        }
        rows_out.append(row)
        print(f"  {r['bearing']}: HI={r['HI']:.2f}  conservative={row['EOL_conservative_s']:.0f}s  "
              f"median={row['EOL_median_s']:.0f}s  q4={row['EOL_gbm_q4_s']:.0f}s  rf={row['EOL_rf_s']:.0f}s")

    test_df = pd.DataFrame(rows_out)
    test_df.to_csv(RESULT_DIR / "28_eol_regressor_test.csv", index=False)

    # Submission xlsx (conservative = 비대칭 페널티 보수)
    sub = test_df[["Bearing", "HI_last", "EOL_conservative_s"]].copy()
    sub.columns = ["Bearing", "HI_last", "RUL_pred_seconds"]
    sub["RUL_pred_hours"] = sub["RUL_pred_seconds"] / 3600.0
    sub.to_excel(RESULT_DIR / "28_eol_regressor_submission.xlsx", index=False)

    print(f"\n  Saved: {RESULT_DIR / '28_eol_regressor_lobo.csv'}")
    print(f"  Saved: {RESULT_DIR / '28_eol_regressor_test.csv'}")
    print(f"  Saved: {RESULT_DIR / '28_eol_regressor_submission.xlsx'}")
    print(f"  Saved models: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
