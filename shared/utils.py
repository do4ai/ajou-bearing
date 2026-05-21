"""공통 유틸리티 — 세 방법론 공유"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

TRAIN_NAMES = ["Train1", "Train2", "Train3", "Train4"]
VAL_NAMES   = ["Val1", "Val2"]
FS = 25600
ORDERS = {"BPFI": 8.40, "BPFO": 5.58, "BSF": 4.68, "FTF": 0.40}


def asym_score(pred, true):
    """
    챌린지 공식 (퍼센트 오차 기준):
      Er = 100 * (ActRUL - PredRUL) / ActRUL    ← 퍼센트(%)
      A  = exp(-ln(0.5) * Er/20)   if Er <= 0  (늦은 예측: Pred > Act)
      A  = exp(+ln(0.5) * Er/50)   if Er >  0  (이른 예측: Pred < Act)

    pred, true: 초 단위 (seconds)
    A ∈ (0, 1], 완벽 = 1.0
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    Er = 100.0 * (true - pred) / (true + 1e-12)  # 퍼센트 오차
    ln_half = np.log(0.5)  # -0.693
    # Er <= 0 (늦은예측): -ln(0.5)*Er/20, Er<0 → arg<0 → A<1
    # Er >  0 (이른예측): +ln(0.5)*Er/50, Er>0 → arg<0 → A<1
    arg_late  = np.clip(-ln_half * Er / 20.0, -50, 0)
    arg_early = np.clip( ln_half * Er / 50.0, -50, 0)
    A = np.where(Er <= 0, np.exp(arg_late), np.exp(arg_early))
    return float(np.mean(A))


def load_bearing(name):
    """베어링 데이터 로드: (signals [N,4,S], operating_df)"""
    d = DATA_DIR / name
    return np.load(d / "vibration.npy"), pd.read_csv(d / "operating.csv")


if __name__ == "__main__":
    # Er = 100*(Act-Pred)/Act
    # 완벽 예측: Er=0 → A=1
    assert abs(asym_score([10000], [10000]) - 1.0) < 1e-6, "완벽 예측 실패"
    # 늦게 20%: Pred=12000, Act=10000 → Er=100*(10000-12000)/10000=-20 → A=exp(-ln(0.5)*(-20)/20)=exp(ln(0.5))=0.5
    assert abs(asym_score([12000], [10000]) - 0.5) < 1e-3, f"늦게20%: {asym_score([12000],[10000])}"
    # 이르게 50%: Pred=5000, Act=10000 → Er=100*(10000-5000)/10000=+50 → A=exp(ln(0.5)*50/50)=exp(ln(0.5))=0.5
    assert abs(asym_score([5000], [10000]) - 0.5) < 1e-3, f"이르게50%: {asym_score([5000],[10000])}"
    print("asym_score 검증 통과 (퍼센트 오차)")
    print(f"  완벽:     {asym_score([10000],[10000]):.4f}")
    print(f"  늦게 20%: {asym_score([12000],[10000]):.4f}  (should be 0.5)")
    print(f"  이른 50%: {asym_score([5000],[10000]):.4f}  (should be 0.5)")
