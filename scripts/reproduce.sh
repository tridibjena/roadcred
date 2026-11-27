#!/usr/bin/env bash
# Reproduce every table and figure in results/RESULTS.md from raw IDD.
#
# Assumes IDD Lite is extracted to data/raw/idd_seg/ (see DOWNLOADS.md). Roughly 14 hours
# on an M1 Pro: 25 training runs of ~15-35 minutes (the stride-8 run ~1 hour), then the
# analyses. Each stage writes its CSV incrementally, so an interrupted run leaves usable
# partial output. Set PY to use an interpreter other than the venv's.
#
# Several suites deliberately repeat the baseline configuration (DeepLabV3+/ResNet-34,
# cross-entropy, seed 0 on IDD's own split): the split, loss, arch and stride suites each
# contain it. RESULTS.md §1c measures run-to-run noise from those repeats.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-.venv/bin/python}

echo "==> 1/5  Dataset variants (official / sequence / sequence3 / frame) and dataset facts"
$PY -m data.prepare_masks --split-mode all --target level1

echo "==> 2/5  Leakage: the split experiment, then three seeds per arm"
$PY scripts/run_ablations.py --suite split --seeds 0
$PY scripts/run_ablations.py --suite seedvar --seeds 0 1 2

echo "==> 3/5  Loss, decoder, initialisation and stride ablations"
$PY scripts/run_ablations.py --suite loss --seeds 0
$PY scripts/run_ablations.py --suite arch --seeds 0
$PY scripts/run_ablations.py --suite pretrain --seeds 0
$PY scripts/run_ablations.py --suite stride --seeds 0

echo "==> 4/5  Held-out test set: train on the 3-way split, score val and the untouched test"
$PY scripts/run_ablations.py --suite heldout --seeds 0
for split in val test; do
  $PY -m evaluation.eval --checkpoint checkpoints/heldout_sequence3_s0.pt \
    --data data/processed/level1_sequence3 --split "$split" --out "results/heldout_${split}.json"
done

echo "==> 5/5  Post-training analysis on the best checkpoint, then RESULTS.md"
PY="$PY" ./scripts/finalize.sh
