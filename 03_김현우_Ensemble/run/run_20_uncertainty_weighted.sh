#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/20_UncertaintyWeighted_TargetAdaptation/experiment.py
