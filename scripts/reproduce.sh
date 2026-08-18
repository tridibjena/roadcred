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

echo "==> 1/7  Building dataset variants (official / sequence / random-frame)"
$PY -m data.prepare_masks --split-mode all --target level1

echo "==> 2/7  Split experiment (the leakage result)"
$PY scripts/run_ablations.py --suite split --seeds 0

echo "==> 3/7  Loss ablation"
$PY scripts/run_ablations.py --suite loss --seeds 0

echo "==> 4/7  Architecture comparison and encoder initialisation"
$PY scripts/run_ablations.py --suite arch --seeds 0
$PY scripts/run_ablations.py --suite pretrain --seeds 0

# Pick the best checkpoint by validation mIoU across every run logged so far.
BEST=$($PY - <<'PYEOF'
import csv, pathlib
rows = [r for r in csv.DictReader(open("results/runs.csv")) if r.get("miou") and r.get("checkpoint")]
rows = [r for r in rows if pathlib.Path(r["checkpoint"]).exists()]
print(max(rows, key=lambda r: float(r["miou"]))["checkpoint"] if rows else "")
PYEOF
)
if [ -z "$BEST" ]; then echo "No checkpoint found; stopping."; exit 1; fi
echo "==> best checkpoint: $BEST"

echo "==> 5/7  Calibration, error analysis, prediction stability"
$PY -m evaluation.calibration   --checkpoint "$BEST"
$PY -m evaluation.error_analysis --checkpoint "$BEST"
$PY -m evaluation.stability      --checkpoint "$BEST"

echo "==> 6/7  Corruption robustness with test-time BatchNorm adaptation"
$PY -m evaluation.corruption_eval --checkpoint "$BEST" --severities 1 3 5 --fgsm

echo "==> 7/7  ONNX export, parity check, INT8 quantization, latency benchmark"
$PY -m compression.quantize --checkpoint "$BEST"

$PY scripts/build_results.py
echo
echo "Done. See results/RESULTS.md"
