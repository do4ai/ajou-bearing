#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/05_HIBlend_Baseline_ChannelSym/blend.py
