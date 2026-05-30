#!/usr/bin/env bash
# Build 팀이름_code.zip — KSPHM-KIMM 2026 제출용 재현 가능 코드 패키지.
#
# 포함: 코드 (experiments, shared, tools, run), 문서, 소형 결과 (csv/xlsx), submission
# 제외: 대형 모델 (.pt ~152MB), raw data (.npy/.tdms), .git, __pycache__
# 모델 재학습 가이드는 README + run/ wrapper로 제공.
#
# Usage: bash tools/build_code_zip.sh [팀명]

set -euo pipefail

TEAM="${1:-HUFS}"
ENSEMBLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAGE_DIR="$(mktemp -d)/${TEAM}_code"
OUT_ZIP="${ENSEMBLE_DIR}/artifacts/submissions/${TEAM}_code.zip"

echo "=================================================="
echo "Building code zip: ${TEAM}_code.zip"
echo "=================================================="

mkdir -p "${STAGE_DIR}"

# 코드 + 문서 복사 (대형 파일 제외)
for d in experiments shared tools run docs; do
  if [ -d "${ENSEMBLE_DIR}/${d}" ]; then
    rsync -a \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.pt' \
      --exclude='*.npy' \
      --exclude='*.npz' \
      --exclude='*.tdms' \
      "${ENSEMBLE_DIR}/${d}" "${STAGE_DIR}/" 2>/dev/null || true
  fi
done

# 루트 문서
for f in README.md VERSION_MAP.md; do
  [ -f "${ENSEMBLE_DIR}/${f}" ] && cp "${ENSEMBLE_DIR}/${f}" "${STAGE_DIR}/"
done

# shared/utils.py (repo 루트의 shared) — 평가식 포함, 필수
REPO_ROOT="$(cd "${ENSEMBLE_DIR}/.." && pwd)"
# shared/는 __init__.py 가진 패키지. ENSEMBLE_DIR이 sys.path에 추가되므로
# STAGE_DIR/shared/ 에 넣으면 `from shared.utils import ...`가 zip에서도 동작.
if [ -f "${REPO_ROOT}/shared/utils.py" ]; then
  mkdir -p "${STAGE_DIR}/shared"
  cp "${REPO_ROOT}/shared/utils.py" "${STAGE_DIR}/shared/utils.py"
  cp "${REPO_ROOT}/shared/__init__.py" "${STAGE_DIR}/shared/__init__.py" 2>/dev/null || touch "${STAGE_DIR}/shared/__init__.py"
fi

# 소형 결과/제출물 (csv, xlsx만; png는 보고용 일부)
mkdir -p "${STAGE_DIR}/artifacts/submissions"
cp "${ENSEMBLE_DIR}/artifacts/submissions/"*.xlsx "${STAGE_DIR}/artifacts/submissions/" 2>/dev/null || true
cp "${ENSEMBLE_DIR}/artifacts/submissions/"*.md "${STAGE_DIR}/artifacts/submissions/" 2>/dev/null || true

mkdir -p "${STAGE_DIR}/artifacts/results/17_AsymOptimal_TrainBased"
cp "${ENSEMBLE_DIR}/artifacts/results/17_AsymOptimal_TrainBased/"*.csv \
   "${STAGE_DIR}/artifacts/results/17_AsymOptimal_TrainBased/" 2>/dev/null || true
cp "${ENSEMBLE_DIR}/artifacts/results/17_AsymOptimal_TrainBased/"*.pdf \
   "${STAGE_DIR}/artifacts/results/17_AsymOptimal_TrainBased/" 2>/dev/null || true

# 핵심 feature CSV (271 cols, train-based 예측의 입력)
mkdir -p "${STAGE_DIR}/artifacts/results/06_Dynamics_DTW_TFTBiLSTM"
cp "${ENSEMBLE_DIR}/artifacts/results/06_Dynamics_DTW_TFTBiLSTM/v25_features_dynamics.csv" \
   "${STAGE_DIR}/artifacts/results/06_Dynamics_DTW_TFTBiLSTM/" 2>/dev/null || true

# rpm-aware order features (28_EOLRegressor 정확 재현에 필수, 4.8MB)
mkdir -p "${STAGE_DIR}/artifacts/results/14_RPMAwareOrderFeatures"
cp "${ENSEMBLE_DIR}/artifacts/results/14_RPMAwareOrderFeatures/"*.csv \
   "${STAGE_DIR}/artifacts/results/14_RPMAwareOrderFeatures/" 2>/dev/null || true

# upstream candidate CSV (sensitivity/hybrid full-chain 재현에 필요)
for d in 05_HIBlend_Baseline_ChannelSym 15_TrajectoryKNN_DTW_RUL \
         16_ScoreAware_CalibratedEnsemble 28_EOLRegressor_Specialist; do
  mkdir -p "${STAGE_DIR}/artifacts/results/${d}"
  cp "${ENSEMBLE_DIR}/artifacts/results/${d}/"*.csv \
     "${STAGE_DIR}/artifacts/results/${d}/" 2>/dev/null || true
done

# requirements.txt 생성
cat > "${STAGE_DIR}/requirements.txt" <<'EOF'
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
torch>=2.0
joblib>=1.3
matplotlib>=3.7
openpyxl>=3.1
EOF

# REPRODUCE.md 생성
cat > "${STAGE_DIR}/REPRODUCE.md" <<'EOF'
# 재현 가이드 — KSPHM-KIMM 2026 Bearing RUL

## 1. 환경
```bash
pip install -r requirements.txt
# PyTorch를 XGBoost보다 먼저 import (OpenMP 충돌 방지)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

## 2. 데이터 준비
- TDMS → npy/csv 변환 후 `data/raw/{Train1..4, Test1..6}/` 배치
- 평가식 등 공통 유틸: `shared/utils.py` (`from shared.utils import asym_score`)

## 3. 학습 (모델 재생성)
대형 모델(.pt)은 zip에서 제외. 아래로 재학습:
```bash
# 베이스 모델
python3 experiments/01_Baseline_1ch_TFTBiLSTM_GPR/pipeline.py
python3 experiments/04_ChannelSym_EOLWeighted/pipeline.py
python3 experiments/06_Dynamics_DTW_TFTBiLSTM/pipeline.py

# train-based 예측 (모델 불필요, feature CSV 기반)
python3 experiments/28_EOLRegressor_Specialist/experiment.py
python3 experiments/17_AsymOptimal_TrainBased/experiment.py
python3 experiments/17_AsymOptimal_TrainBased/eol_progression_robust.py
python3 experiments/17_AsymOptimal_TrainBased/sensitivity.py
python3 experiments/17_AsymOptimal_TrainBased/per_bearing_robust.py
python3 experiments/17_AsymOptimal_TrainBased/finalize_submissions.py
```

## 4. 최종 제출 파일
- `artifacts/submissions/HUFS_validation_1순위.xlsx` (Per-Bearing Robust)
- `artifacts/submissions/HUFS_validation_백업1.xlsx` (5_HIBlend)
- `artifacts/submissions/HUFS_validation_백업2.xlsx` (19_EOLProgression)

## 5. 핵심 원칙
- 모든 RUL 출력은 train data로 학습된 회귀값 (임의 clamp 없음)
- 600s 물리 하한만 적용 (측정 간격)
- 비대칭 페널티 (늦은 예측 2.5×)를 학습·예측·calibration에 일관 반영

자세한 방법론: `README.md`, `VERSION_MAP.md`, `artifacts/submissions/PRESUBMISSION_CHECKLIST.md`
EOF

# zip 생성
mkdir -p "${ENSEMBLE_DIR}/artifacts/submissions"
( cd "$(dirname "${STAGE_DIR}")" && zip -rq "${OUT_ZIP}" "$(basename "${STAGE_DIR}")" )

echo ""
echo "✅ Created: ${OUT_ZIP}"
ls -lh "${OUT_ZIP}"
echo ""
echo "Contents summary:"
unzip -l "${OUT_ZIP}" | tail -5
echo ""
echo "Python files included: $(unzip -l "${OUT_ZIP}" | grep -c '\.py$' || true)"

# 임시 디렉토리 정리
rm -rf "$(dirname "${STAGE_DIR}")"
