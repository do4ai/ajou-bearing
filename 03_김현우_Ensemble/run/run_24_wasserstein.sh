#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/24_Wasserstein_SourceWeighted_RUL/experiment.py
