#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/15_TrajectoryKNN_DTW_RUL/experiment.py
