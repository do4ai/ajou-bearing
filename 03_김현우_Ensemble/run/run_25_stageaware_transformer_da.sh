#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/25_StageAwareTransformer_DA/experiment.py
