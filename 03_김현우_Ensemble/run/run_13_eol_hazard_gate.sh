#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/13_EOLHazardGate_Calibrator/experiment.py
