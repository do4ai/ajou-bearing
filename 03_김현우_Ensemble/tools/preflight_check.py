"""preflight_check — 제출 직전 최종 게이트.

3대 필수 제출물(validation xlsx / code zip / report pdf) 존재·유효성·일관성 검증.
재실행 가능. manifest(파일 크기 + 체크섬 + 검증결과) 생성.

Usage: python3 tools/preflight_check.py [팀명]
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import hashlib
from pathlib import Path

import pandas as pd

ENSEMBLE = Path(__file__).resolve().parents[1]
SUB = ENSEMBLE / "artifacts" / "submissions"
TEAM = sys.argv[1] if len(sys.argv) > 1 else "HUFS"
VAL_NAMES = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]


def sha8(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:8]


def check_validation(fp: Path) -> tuple[bool, str]:
    if not fp.exists():
        return False, "파일 없음"
    df = pd.read_excel(fp)
    issues = []
    if "Bearing" not in df.columns: issues.append("Bearing 컬럼 없음")
    if "RUL_pred_seconds" not in df.columns: issues.append("RUL_pred_seconds 컬럼 없음")
    if len(df) != 6: issues.append(f"행 {len(df)}≠6")
    if "Bearing" in df.columns and sorted(df["Bearing"]) != sorted(VAL_NAMES):
        issues.append("베어링 이름 불일치")
    if "RUL_pred_seconds" in df.columns:
        r = df["RUL_pred_seconds"]
        if r.isna().any(): issues.append("NaN 존재")
        if (r < 600).any(): issues.append("RUL<600 존재")
        if (r > 200000).any(): issues.append("RUL>200000 비현실")
    return (len(issues) == 0), ("OK" if not issues else "; ".join(issues))


def main() -> None:
    print("=" * 70)
    print(f"Pre-flight Check — 팀 {TEAM}")
    print("=" * 70)

    manifest = []
    all_ok = True

    # 1. Validation xlsx (필수 1순위 + 백업들)
    print("\n[1] Validation 제출물")
    val_files = sorted(SUB.glob(f"{TEAM}_validation_*.xlsx"))
    if not val_files:
        print("  ✗ validation xlsx 없음"); all_ok = False
    for fp in val_files:
        ok, msg = check_validation(fp)
        all_ok = all_ok and ok
        sz = fp.stat().st_size
        print(f"  {'✓' if ok else '✗'} {fp.name}: {msg} ({sz}B, sha:{sha8(fp)})")
        manifest.append((fp.name, sz, sha8(fp), msg))

    # 2. Code zip (필수)
    print("\n[2] Code 제출물")
    zip_fp = SUB / f"{TEAM}_code.zip"
    if zip_fp.exists():
        import zipfile
        with zipfile.ZipFile(zip_fp) as z:
            names = z.namelist()
            py_cnt = sum(1 for n in names if n.endswith(".py"))
            has_repro = any("REPRODUCE.md" in n for n in names)
            has_shared = any(n.endswith("shared/utils.py") for n in names)
            has_feat = any("v25_features_dynamics.csv" in n for n in names)
        ok = py_cnt > 0 and has_repro and has_shared and has_feat
        all_ok = all_ok and ok
        print(f"  {'✓' if ok else '✗'} {zip_fp.name}: {len(names)} files, {py_cnt} py, "
              f"REPRODUCE={has_repro}, shared/utils={has_shared}, feat_csv={has_feat} "
              f"({zip_fp.stat().st_size//1024}KB)")
        manifest.append((zip_fp.name, zip_fp.stat().st_size, sha8(zip_fp),
                          f"{len(names)}files/{py_cnt}py"))
    else:
        print("  ✗ code.zip 없음"); all_ok = False

    # 3. Report pdf (필수)
    print("\n[3] Report 제출물")
    pdf_fp = SUB / f"{TEAM}_report.pdf"
    if pdf_fp.exists():
        sz = pdf_fp.stat().st_size
        head = pdf_fp.read_bytes()[:5]
        ok = head.startswith(b"%PDF")
        all_ok = all_ok and ok
        print(f"  {'✓' if ok else '✗'} {pdf_fp.name}: PDF헤더={ok} ({sz//1024}KB, sha:{sha8(pdf_fp)})")
        manifest.append((pdf_fp.name, sz, sha8(pdf_fp), "PDF" if ok else "헤더오류"))
    else:
        print("  ✗ report.pdf 없음"); all_ok = False

    # 4. 모든 후보 ↔ source 일관성 (4 제출 후보가 각 문서화된 메서드와 bit-exact 일치하는지)
    print("\n[4] 후보 ↔ source 일관성 (4종 모두)")
    R = ENSEMBLE / "artifacts/results"
    cand_src = [
        ("pstar", "17_AsymOptimal_TrainBased/37_pstar_submission.xlsx"),                       # flagship (재-base)
        ("pstar_conservative", "17_AsymOptimal_TrainBased/37_pstar_conservative.xlsx"),         # flagship × β0.97
        ("1순위", "17_AsymOptimal_TrainBased/18_per_bearing_robust_submission.xlsx"),           # 대조군(메타-셀렉터)
        ("1순위_conservative", "17_AsymOptimal_TrainBased/23_per_bearing_beta095_submission.xlsx"),
        ("백업1", "05_HIBlend_Baseline_ChannelSym/submission_v24_v17v22_combined.xlsx"),
        ("백업2", "17_AsymOptimal_TrainBased/19_eol_progression_robust_submission.xlsx"),
        ("finaltest_robust", "26_FinalRobust_LOBOFrozenSelector/26_final_robust_submission.xlsx"),
    ]
    order = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6"]
    for nm, rel in cand_src:
        pc = SUB / f"{TEAM}_validation_{nm}.xlsx"
        src = R / rel
        if pc.exists() and src.exists():
            a = pd.read_excel(pc).set_index("Bearing")["RUL_pred_seconds"].reindex(order)
            b = pd.read_excel(src).set_index("Bearing")["RUL_pred_seconds"].reindex(order)
            diff = float((a - b).abs().max())
            ok = diff < 1
            all_ok = all_ok and ok
            print(f"  {'✓' if ok else '✗'} {nm:20s} vs {rel.split('/')[-1]:42s} max diff={diff:.2f}")
        else:
            print(f"  - {nm}: source 비교 불가 (파일 누락)")

    # Manifest 저장
    man_fp = SUB / "PREFLIGHT_MANIFEST.txt"
    with open(man_fp, "w") as f:
        f.write(f"Pre-flight Manifest — 팀 {TEAM}\n")
        f.write(f"전체 상태: {'PASS' if all_ok else 'FAIL'}\n\n")
        f.write(f"{'파일':<45} {'크기':>10} {'sha':>10}  결과\n")
        for name, sz, sha, msg in manifest:
            f.write(f"{name:<45} {sz:>10} {sha:>10}  {msg}\n")

    print("\n" + "=" * 70)
    print(f"  전체 상태: {'✓ PASS — 제출 가능' if all_ok else '✗ FAIL — 문제 해결 필요'}")
    print(f"  Manifest: {man_fp}")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
