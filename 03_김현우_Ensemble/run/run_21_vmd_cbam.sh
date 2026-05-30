#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/21_VMD_CBAM_FeatureDenoising/experiment.py
