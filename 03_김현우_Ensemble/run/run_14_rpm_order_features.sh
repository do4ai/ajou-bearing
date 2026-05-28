#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/14_RPMAwareOrderFeatures/experiment.py
