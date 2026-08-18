#!/usr/bin/env bash
# Reproduce every table and figure in results/RESULTS.md from raw IDD.
#
# Assumes IDD Lite is extracted to data/raw/idd_seg/ (see DOWNLOADS.md) and the venv
# exists. Total runtime is roughly 7 hours on an M1 Pro; each stage writes its CSV
# incrementally, so an interrupted run leaves usable partial output.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python

echo "==> 1/5  Building dataset variants (official / sequence / random-frame)"
$PY -m data.prepare_masks --split-mode all --target level1

echo "==> 2/5  Split experiment (the leakage result)"
$PY scripts/run_ablations.py --suite split --seeds 0

echo "==> 3/5  Loss ablation"
$PY scripts/run_ablations.py --suite loss --seeds 0

echo "==> 4/5  Architecture comparison and encoder initialisation"
$PY scripts/run_ablations.py --suite arch --seeds 0
$PY scripts/run_ablations.py --suite pretrain --seeds 0

echo "==> 5/5  Post-training analysis (calibration, robustness, distillation, Pareto)"
./scripts/finalize.sh
