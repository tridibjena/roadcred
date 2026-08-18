#!/usr/bin/env bash
# Run the experiment suites back to back, waiting for any in-flight run first.
# Keeps the GPU busy while the rest of the project is being written.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python

# Wait for an already-running suite (e.g. the split experiment) to finish.
while pgrep -f "run_ablations.py" > /dev/null 2>&1; do sleep 20; done

echo "=== loss ablation (5 losses x 1 seed) ==="
$PY scripts/run_ablations.py --suite loss --seeds 0 --epochs 30 --patience 8 >> /tmp/loss.log 2>&1

echo "=== architecture comparison ==="
$PY scripts/run_ablations.py --suite arch --seeds 0 --epochs 30 --patience 8 >> /tmp/arch.log 2>&1

echo "=== encoder initialisation ==="
$PY scripts/run_ablations.py --suite pretrain --seeds 0 --epochs 30 --patience 8 >> /tmp/pretrain.log 2>&1

echo "=== all suites complete ==="
