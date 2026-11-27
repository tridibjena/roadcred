#!/usr/bin/env bash
# Post-training analysis: everything that needs a trained model.
# Waits for any in-flight training suite, picks the best checkpoint by validation mIoU on
# IDD's own split, then runs calibration, error analysis, stability, robustness (with
# FGSM), calibration under shift, distillation, the INT8 export the API serves, the
# road-edge caveat check and the deployment Pareto, and regenerates RESULTS.md.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-.venv/bin/python}

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
# Only runs validated on IDD's own split are comparable with each other -- and only they
# are safe to analyse on it: the pooled re-splits (sequence, frame) train on frames from
# IDD's own validation set, so every analysis below would score them on training data.
rows = [r for r in rows if r.get("data") == "level1_official"
        and r.get("checkpoint") and pathlib.Path(r["checkpoint"]).exists()]
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
echo "==> corruption robustness (all 10, severities 1/3) + FGSM"
$PY -m evaluation.corruption_eval --checkpoint "$BEST" --severities 1 3 --fgsm || echo "  (failed)"
echo "==> calibration under shift"
$PY -m evaluation.shift_calibration --checkpoint "$BEST" \
  --corruptions fog rain low_light motion_blur gaussian_noise || echo "  (failed)"

echo "==> distillation: teacher -> resnet18 student, both arms on one recipe"
$PY -m compression.distill --teacher "$BEST" --student-encoder resnet18 || echo "  (failed)"

echo "==> INT8 export of the best checkpoint (the model the API serves)"
$PY -m compression.quantize --checkpoint "$BEST" || echo "  (failed)"
echo "==> road-edge caveat check (served model + its calibration)"
$PY -m evaluation.road_edge_caveat || echo "  (failed)"

# Latency is measured on CPU: run this on an otherwise idle machine, or the timings drift.
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
