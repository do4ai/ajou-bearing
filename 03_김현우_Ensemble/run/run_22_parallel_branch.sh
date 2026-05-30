#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/22_ParallelTCNTransformer_Branch/experiment.py
