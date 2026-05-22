"""
TDMS → numpy+CSV 변환 스크립트
KIMM 베어링 열화시험 데이터 (KSPHM-KIMM 2026 챌린지)

사용법:
  python convert_tdms.py --input /path/to/extracted_zip --output ../raw

데이터 구조 (예상):
  Train.zip → Train1/, Train2/, Train3/, Train4/
  Test.zip  → Val1/, Val2/  (또는 Test1/, Test2/)

각 베어링 폴더에 여러 TDMS 파일이 있을 수 있음.
변환 후: data/raw/{Train1,...,Val2}/vibration.npy, operating.csv
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from nptdms import TdmsFile
import re


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def explore_tdms(tdms_path):
    """TDMS 파일 구조 탐색 → 채널 목록과 메타데이터 반환"""
    tf = TdmsFile.read(tdms_path)
    info = {}
    for group in tf.groups():
        gname = group.name
        info[gname] = {}
        for ch in group.channels():
            cname = ch.name
            data = ch.data
            props = ch.properties
            info[gname][cname] = {
                "dtype": str(data.dtype) if len(data) > 0 else "empty",
                "length": len(data),
                "properties": {k: str(v) for k, v in props.items()},
            }
    return info


def read_tdms(tdms_path):
    """TDMS 파일에서 진동+운전조건 데이터 추출"""
    tf = TdmsFile.read(tdms_path)
    signals = {}
    operating = {}

    for group in tf.groups():
        for ch in group.channels():
            name_lower = ch.name.lower()
            data = ch.data

            if len(data) == 0:
                continue

            # 진동 채널 감지 (길이가 긴 데이터)
            if len(data) > 1000:
                signals[ch.name] = data
            # 운전조건 채널 감지 (RPM, Torque, Temp 등)
            elif any(kw in name_lower for kw in ["rpm", "speed", "torque", "temp", "force", "load", "power"]):
                operating[ch.name] = float(np.nanmean(data)) if len(data) > 0 else np.nan

    return signals, operating


def convert_bearing(bearing_dir, output_dir, bearing_name):
    """단일 베어링의 TDMS 파일들을 변환"""
    tdms_files = sorted(Path(bearing_dir).glob("**/*.tdms"), key=lambda p: natural_sort_key(str(p)))

    if not tdms_files:
        print(f"  ⚠ {bearing_name}: TDMS 파일 없음 in {bearing_dir}")
        return False

    print(f"  {bearing_name}: {len(tdms_files)}개 TDMS 파일")

    # 첫 번째 파일로 구조 탐색
    if len(tdms_files) <= 3:
        for f in tdms_files[:3]:
            info = explore_tdms(f)
            print(f"\n  구조 ({f.name}):")
            for gname, channels in info.items():
                print(f"    Group: {gname}")
                for cname, meta in channels.items():
                    print(f"      {cname}: len={meta['length']}, dtype={meta['dtype']}")
                    if meta['properties']:
                        for pk, pv in list(meta['properties'].items())[:5]:
                            print(f"        {pk}={pv}")
    else:
        info = explore_tdms(tdms_files[0])
        print(f"\n  구조 예시 ({tdms_files[0].name}):")
        for gname, channels in info.items():
            print(f"    Group: {gname}")
            for cname, meta in channels.items():
                print(f"      {cname}: len={meta['length']}, dtype={meta['dtype']}")

    all_signals = []
    all_operating = []

    for i, tdms_path in enumerate(tdms_files):
        signals, operating = read_tdms(tdms_path)

        if not signals:
            continue

        # 진동 데이터를 [4, samples] 형태로 구성
        sig_keys = sorted(signals.keys())
        sig_array = np.array([signals[k] for k in sig_keys], dtype=np.float32)

        # 채널 수가 다르면 최대 4채널로 맞춤
        if sig_array.shape[0] > 4:
            sig_array = sig_array[:4]
        elif sig_array.shape[0] < 4:
            pad = np.zeros((4 - sig_array.shape[0], sig_array.shape[1]), dtype=np.float32)
            sig_array = np.vstack([sig_array, pad])

        all_signals.append(sig_array)

        # 운전조건 수집
        op = {"measurement": i}
        op.update(operating)
        all_operating.append(op)

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(tdms_files)}")

    if not all_signals:
        print(f"  ⚠ {bearing_name}: 변환할 데이터 없음")
        return False

    # signals: [N, 4, S] 형태
    sigs_np = np.array(all_signals, dtype=np.float32)
    print(f"  진동 shape: {sigs_np.shape}")

    # operating DataFrame
    op_df = pd.DataFrame(all_operating)

    # 컬럼명 표준화
    col_map = {}
    for c in op_df.columns:
        cl = c.lower()
        if "rpm" in cl or "speed" in cl:
            col_map[c] = "rpm"
        elif "torque" in cl or "force" in cl:
            col_map[c] = "torque"
        elif "front" in cl or "out" in cl or "outer" in cl:
            col_map[c] = "temp_front"
        elif "rear" in cl or "in" in cl or "inner" in cl:
            col_map[c] = "temp_rear"
    op_df = op_df.rename(columns=col_map)

    # 시간 정보 계산 (10분 간격, 1분 취득)
    n = len(op_df)
    interval = 600  # 10분 = 600초
    op_df["t_seconds"] = np.arange(n) * interval
    op_df["t_seconds"] = op_df["t_seconds"].astype(float)

    # EOL 시간 (마지막 측정 + interval)
    eol = op_df["t_seconds"].iloc[-1] + interval

    # RUL 계산
    op_df["rul_seconds"] = (eol - op_df["t_seconds"]).astype(float)
    op_df["rul_seconds"] = op_df["rul_seconds"].clip(lower=0)

    # 저장
    out_path = Path(output_dir) / bearing_name
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "vibration.npy", sigs_np)
    op_df.to_csv(out_path / "operating.csv", index=False)

    print(f"  저장: {out_path}/")
    print(f"    vibration.npy: {sigs_np.shape} ({sigs_np.nbytes / 1e6:.1f} MB)")
    print(f"    operating.csv: {len(op_df)} rows, 컬럼={list(op_df.columns)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="TDMS → numpy+CSV 변환")
    parser.add_argument("--input", required=True, help="압축 해제된 데이터 폴더 경로")
    parser.add_argument("--output", default=None, help="출력 경로 (기본: data/raw/)")
    parser.add_argument("--explore", action="store_true", help="TDMS 구조만 탐색")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(__file__).parent / "raw"

    if args.explore:
        tdms_files = sorted(input_path.glob("**/*.tdms"), key=lambda p: natural_sort_key(str(p)))
        if not tdms_files:
            print(f"TDMS 파일 없음 in {input_path}")
            return
        print(f"TDMS 파일 {len(tdms_files)}개 발견")
        for f in tdms_files[:3]:
            print(f"\n{'='*60}")
            print(f"파일: {f.name}")
            info = explore_tdms(f)
            for gname, channels in info.items():
                print(f"  Group: {gname}")
                for cname, meta in channels.items():
                    props_str = ", ".join(f"{k}={v}" for k, v in list(meta['properties'].items())[:3])
                    print(f"    {cname}: len={meta['length']}, dtype={meta['dtype']}, props=[{props_str}]")
        return

    # 베어링 폴더 자동 감지
    bearing_dirs = {}
    for pattern in ["Train*", "train*", "Val*", "val*", "Test*", "test*"]:
        for d in sorted(input_path.glob(pattern), key=lambda p: natural_sort_key(str(p))):
            if d.is_dir():
                name = d.name
                # 표준 이름으로 변환
                if name.lower().startswith("train"):
                    num = re.search(r'\d+', name)
                    bearing_dirs[f"Train{num.group()}" if num else name] = d
                elif name.lower().startswith("val"):
                    num = re.search(r'\d+', name)
                    bearing_dirs[f"Val{num.group()}" if num else name] = d
                elif name.lower().startswith("test"):
                    num = re.search(r'\d+', name)
                    bearing_dirs[f"Val{num.group()}" if num else name] = d

    # 하위 폴더에 TDMS가 있으면 포함
    if not bearing_dirs:
        # 직접 TDMS 파일이 있는지 확인
        tdms_files = list(input_path.glob("*.tdms"))
        if tdms_files:
            bearing_dirs[input_path.name] = input_path

    if not bearing_dirs:
        # 더 깊이 탐색
        for d in sorted(input_path.iterdir()):
            if d.is_dir():
                tdms_in = list(d.glob("**/*.tdms"))
                if tdms_in:
                    bearing_dirs[d.name] = d

    print(f"베어링 폴더 {len(bearing_dirs)}개: {list(bearing_dirs.keys())}")

    success = 0
    for name, dir_path in bearing_dirs.items():
        print(f"\n{'='*60}")
        print(f"변환: {name} ({dir_path})")
        if convert_bearing(dir_path, output_path, name):
            success += 1

    print(f"\n{'='*60}")
    print(f"완료: {success}/{len(bearing_dirs)} 베어링 변환 성공")
    print(f"출력: {output_path}")


if __name__ == "__main__":
    main()
