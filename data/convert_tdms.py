"""
TDMS → numpy+CSV 변환 (KIMM 베어링 챌린지 실제 데이터)

데이터 구조:
  download/extracted/
    Train{n}_Vibration/000001.tdms ...  (Vibration 그룹 / CH1~CH4 / 1.5M sample × float32)
    Train{n}_Operation.csv              (Time[sec], Torque, Motor speed, TC Front, TC Rear @ 10s)
    Test{n}/000001.tdms ...             (Test는 운전조건 비공개)

샘플링:
  - 진동 25.6kHz × 4채널, 60초/측정 (1,536,000 samples), 10분 주기
  - 운전조건 0.1Hz

출력:
  data/raw/{Train1..Train4, Test1..Test6}/
    vibration.npy : [N, 4, SAMPLES_PER_FILE]  (기본 2초 다운샘플 = 51200)
    operating.csv : t_seconds, rpm, torque, temp_front, temp_rear, rul_seconds

사용:
  python convert_tdms.py                       # 전체 변환
  python convert_tdms.py --keep-tdms           # 변환 후 TDMS 보존 (기본은 삭제)
  python convert_tdms.py --samples 102400      # 4초 윈도우로 변환
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from nptdms import TdmsFile

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "download" / "extracted"
OUT_DIR = ROOT / "data" / "raw"

MEAS_INTERVAL = 600        # 10분 주기 (초)
MEAS_LENGTH = 60           # 1분 취득 (초)
FS = 25600                 # 25.6 kHz
N_CHANNELS = 4


def read_tdms_vibration(tdms_path, n_samples):
    """TDMS → [4, n_samples] (앞 n_samples만 사용)"""
    tf = TdmsFile.read(tdms_path)
    grp = tf["Vibration"]
    chs = [f"CH{i}" for i in (1, 2, 3, 4)]
    sigs = np.stack([np.asarray(grp[c].data[:n_samples], dtype=np.float32) for c in chs])
    return sigs


def operation_for_measurement(op_df, t_start, t_end):
    """[t_start, t_end] 구간의 운전조건 평균(없으면 가장 가까운 값)"""
    sub = op_df[(op_df["Time[sec]"] >= t_start) & (op_df["Time[sec]"] <= t_end + 5)]
    if len(sub) == 0:
        # 가장 가까운 한 행
        idx = (op_df["Time[sec]"] - t_start).abs().idxmin()
        sub = op_df.iloc[[idx]]
    return {
        "rpm": float(sub.iloc[:, 2].mean()),
        "torque": float(sub.iloc[:, 1].mean()),
        "temp_front": float(sub.iloc[:, 3].mean()),
        "temp_rear": float(sub.iloc[:, 4].mean()),
    }


def convert_train(name, src_dir, op_csv, out_dir, n_samples):
    tdms_files = sorted(src_dir.glob("*.tdms"))
    if not tdms_files:
        print(f"  ✗ {name}: TDMS 파일 없음")
        return False
    N = len(tdms_files)
    print(f"  {name}: {N} measurements, sampling 0~{n_samples/FS:.2f}s of each {MEAS_LENGTH}s capture")

    try:
        op_df = pd.read_csv(op_csv, encoding="cp949")
    except UnicodeDecodeError:
        op_df = pd.read_csv(op_csv, encoding="latin-1")
    op_df.columns = [c.strip() for c in op_df.columns]
    # 컬럼명이 깨질 수 있어 인덱스 기반 접근
    op_df.rename(columns={op_df.columns[0]: "Time[sec]"}, inplace=True)

    sigs = np.empty((N, N_CHANNELS, n_samples), dtype=np.float32)
    rows = []
    for i, tdms in enumerate(tdms_files):
        sigs[i] = read_tdms_vibration(tdms, n_samples)
        t_start = i * MEAS_INTERVAL
        op = operation_for_measurement(op_df, t_start, t_start + MEAS_LENGTH)
        rows.append({
            "measurement": i,
            "t_seconds": float(t_start),
            "rul_seconds": float((N - i) * MEAS_INTERVAL),
            **op,
        })
        if (i + 1) % 25 == 0 or i == N - 1:
            print(f"    {i+1}/{N}", flush=True)

    op_out = pd.DataFrame(rows)
    out_path = out_dir / name
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "vibration.npy", sigs)
    op_out.to_csv(out_path / "operating.csv", index=False)
    print(f"    → {out_path}/  vibration.npy {sigs.shape} ({sigs.nbytes/1e6:.1f}MB), operating.csv {len(op_out)} rows")
    return True


def convert_test(name, src_dir, out_dir, n_samples):
    tdms_files = sorted(src_dir.glob("*.tdms"))
    if not tdms_files:
        print(f"  ✗ {name}: TDMS 파일 없음")
        return False
    N = len(tdms_files)
    print(f"  {name}: {N} measurements (Test, no operation data)")

    sigs = np.empty((N, N_CHANNELS, n_samples), dtype=np.float32)
    rows = []
    for i, tdms in enumerate(tdms_files):
        sigs[i] = read_tdms_vibration(tdms, n_samples)
        t_start = i * MEAS_INTERVAL
        rows.append({
            "measurement": i,
            "t_seconds": float(t_start),
            "rul_seconds": float("nan"),
            "rpm": float("nan"),
            "torque": float("nan"),
            "temp_front": float("nan"),
            "temp_rear": float("nan"),
        })

    op_out = pd.DataFrame(rows)
    out_path = out_dir / name
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "vibration.npy", sigs)
    op_out.to_csv(out_path / "operating.csv", index=False)
    print(f"    → {out_path}/  vibration.npy {sigs.shape} ({sigs.nbytes/1e6:.1f}MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR), help="압축 해제된 폴더")
    ap.add_argument("--out", default=str(OUT_DIR), help="출력 폴더")
    ap.add_argument("--samples", type=int, default=51200,
                    help="측정 1회당 추출 sample 수 (기본 2초=51200)")
    ap.add_argument("--keep-tdms", action="store_true",
                    help="변환 후 원본 TDMS 폴더 보존 (기본은 삭제로 디스크 절약)")
    ap.add_argument("--only", default="", help="콤마구분: Train1,Test3 처럼 일부만")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    only = set([s.strip() for s in args.only.split(",") if s.strip()])

    print(f"src={src}  out={out}  samples={args.samples}  keep-tdms={args.keep_tdms}")
    print(f"{'='*65}")

    # Train1~4
    for n in (1, 2, 3, 4):
        name = f"Train{n}"
        if only and name not in only:
            continue
        src_dir = src / f"Train{n}_Vibration"
        op_csv = src / f"Train{n}_Operation.csv"
        if not src_dir.exists():
            print(f"  ✗ {name}: {src_dir} 없음")
            continue
        ok = convert_train(name, src_dir, op_csv, out, args.samples)
        if ok and not args.keep_tdms:
            shutil.rmtree(src_dir)
            op_csv.unlink(missing_ok=True)
            print(f"    (정리: {src_dir.name} 삭제)")

    # Test1~6
    for n in (1, 2, 3, 4, 5, 6):
        name = f"Test{n}"
        if only and name not in only:
            continue
        src_dir = src / f"Test{n}"
        if not src_dir.exists():
            print(f"  ✗ {name}: {src_dir} 없음")
            continue
        ok = convert_test(name, src_dir, out, args.samples)
        if ok and not args.keep_tdms:
            shutil.rmtree(src_dir)
            print(f"    (정리: {src_dir.name} 삭제)")

    print(f"\n{'='*65}\n완료. 출력: {out}")


if __name__ == "__main__":
    main()
