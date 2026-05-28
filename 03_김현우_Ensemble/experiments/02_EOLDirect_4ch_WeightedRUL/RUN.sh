#!/bin/bash
# v18 학습 완료 후 자동 실행: 후처리 + 다양한 전략 inference

cd "$(dirname "$0")"

echo "=== 1) v18 후처리 (전략 + β + HI-shrinkage 그리드서치) ==="
python3 postprocess_v18_final.py 2>&1 | tee results/v18_postprocess.log

echo ""
echo "=== 2) 기본 inference (median) ==="
python3 infer_test_v18.py --strategy median 2>&1 | tee results/v18_infer_median.log

echo ""
echo "=== 3) trim inference ==="
python3 infer_test_v18.py --strategy trim 2>&1 | tee results/v18_infer_trim.log

echo ""
echo "=== 4) HI-shrinkage with thr=0.85 ==="
python3 infer_test_v18.py --strategy median --hi_thr 0.85 --hi_slope 20 2>&1 | tee results/v18_infer_himed.log

echo ""
echo "=== DONE ==="
ls -la results/submission_v18*.xlsx
