#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/27_EOLClassifier_LOBOCalibrated/experiment.py
