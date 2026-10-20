#!/usr/bin/env bash
# Smoke test of the full workflow on synthetic data (a few minutes on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m marspinn run --config configs/quick.yaml "$@"
echo "Results: results/quick (figures/, tables/, results_summary.md)"
