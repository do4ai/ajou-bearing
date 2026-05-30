#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/19_DomainSpecificResidual_Calibrator/experiment.py
