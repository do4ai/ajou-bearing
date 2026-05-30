#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/17_ConditionalMADA_StageAlignment/experiment.py
