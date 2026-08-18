#!/usr/bin/env bash
# Post-training analysis: everything that needs a trained model.
# Waits for any in-flight training suite, picks the best checkpoint by validation mIoU,
# then runs calibration, error analysis, stability, robustness, distillation, and the
# compression Pareto, and regenerates RESULTS.md.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python

# Wait for the whole queue, not just the current run: queue_experiments.sh has gaps
# between suites where no run_ablations.py process exists, and polling only for that
# would let this fire in one of the gaps and analyse a half-finished set of runs.
while pgrep -f "queue_experiments.sh" > /dev/null 2>&1 \
   || pgrep -f "run_ablations.py"     > /dev/null 2>&1; do
  sleep 30
done
echo "training queue drained"

BEST=$($PY - <<'PYEOF'
import csv, pathlib
try:
    rows = [r for r in csv.DictReader(open("results/runs.csv")) if r.get("miou")]
except FileNotFoundError:
    rows = []
rows = [r for r in rows if r.get("checkpoint") and pathlib.Path(r["checkpoint"]).exists()]
print(max(rows, key=lambda r: float(r["miou"]))["checkpoint"] if rows else "")
PYEOF
)
if [ -z "$BEST" ]; then echo "No checkpoint found; stopping."; exit 1; fi
echo "best checkpoint: $BEST"

echo "==> calibration"
$PY -m evaluation.calibration --checkpoint "$BEST" || echo "  (failed)"
echo "==> error analysis"
$PY -m evaluation.error_analysis --checkpoint "$BEST" || echo "  (failed)"
echo "==> prediction stability"
$PY -m evaluation.stability --checkpoint "$BEST" || echo "  (failed)"
echo "==> corruption robustness (all 10, severities 1/3/5) + FGSM"
$PY -m evaluation.corruption_eval --checkpoint "$BEST" --severities 1 3 5 --fgsm || echo "  (failed)"

echo "==> distillation: teacher -> resnet18 student"
$PY -m compression.distill --teacher "$BEST" --student-encoder resnet18 --epochs 30 || echo "  (failed)"

echo "==> compression Pareto"
ALONE=checkpoints/student_alone_resnet18_s0.pt
DISTILLED=checkpoints/student_distilled_resnet18_s0.pt
ARGS="teacher=$BEST"
[ -f "$ALONE" ]     && ARGS="$ARGS student_alone=$ALONE"
[ -f "$DISTILLED" ] && ARGS="$ARGS student_distilled=$DISTILLED"
# shellcheck disable=SC2086
$PY -m compression.pareto --checkpoints $ARGS || echo "  (failed)"

echo "==> regenerating RESULTS.md"
$PY scripts/build_results.py
echo
echo "Done. See results/RESULTS.md"
