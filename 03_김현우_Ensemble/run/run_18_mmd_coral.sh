#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/18_MMD_CORAL_SourceWeighting/experiment.py
