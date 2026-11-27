#!/usr/bin/env bash
# Run one stage of the full pipeline on Kaggle (or any Linux machine with CUDA GPUs).
#
#   bash scripts/kaggle_stage.sh setup   [IDD]   install, fetch weights, link data, run tests
#   bash scripts/kaggle_stage.sh smoke           time two epochs; nothing is kept
#   bash scripts/kaggle_stage.sh train1          data variants + split, seedvar, loss suites
#   bash scripts/kaggle_stage.sh train2  [PREV]  arch, pretrain, stride and held-out suites
#   bash scripts/kaggle_stage.sh analyse [PREV]  finalize.sh: every analysis and RESULTS.md
#
# Every Kaggle session is a fresh container, so each one starts with the clone and
# `setup`, then runs one stage:
#
#   !git clone --branch kaggle-run-1 https://github.com/tridibjena/roadcred.git
#   %cd roadcred
#   !bash scripts/kaggle_stage.sh setup
#   !bash scripts/kaggle_stage.sh train1
#
# IDD  is the IDD Lite archive or extracted folder; searched for under /kaggle/input if
#      omitted. Upload it as a private dataset and attach it to every session.
# PREV is the previous stage's notebook output, attached as input. Its results/ and
#      checkpoints/ are restored first. Omit it when that stage ran in this session.
#
# Each stage is sized to finish well inside one free session (12 h; some sources say 9).
# Training suites are spread across every visible GPU, one process per GPU, and no two
# processes share a suite, so their CSVs never collide; results/runs.csv, which they
# do share, is locked by modeling/tracking.py.
#
# Environment: WORKERS sets dataloader workers per process (default 0, the recipe of
# every reported run); PY sets the interpreter (default python).
set -euo pipefail

PY=${PY:-python}
WORKERS=${WORKERS:-0}

on_kaggle() { [ -d /kaggle/working ]; }

link_data() {
  # DOWNLOADS.md's layout: data/raw/idd_seg/idd20k_lite/{leftImg8bit,gtFine}.
  local target=data/raw/idd_seg/idd20k_lite source="${1:-}"
  if [ -d "$target" ]; then echo "data: $target present"; return; fi
  mkdir -p data/raw/idd_seg
  if [ -z "$source" ] && [ -d /kaggle/input ]; then
    # Kaggle may extract an uploaded archive into folders, or keep it whole.
    source=$(find /kaggle/input -maxdepth 6 -type d -name idd20k_lite -print -quit)
    [ -n "$source" ] || source=$(find /kaggle/input -maxdepth 6 -name 'idd-lite.tar.gz' -print -quit)
  fi
  case "$source" in
    "") echo "IDD Lite not found: attach the dataset, or pass its archive or folder" >&2
        exit 1 ;;
    *.tar.gz|*.tgz) tar -xzf "$source" -C data/raw/idd_seg ;;
    *) [ -d "$source/idd20k_lite" ] && source="$source/idd20k_lite"
       cp -r "$source" "$target" ;;  # a copy, not a link: /kaggle/input is read-only
  esac
  [ -d "$target" ] || { echo "no idd20k_lite/ found in $source" >&2; exit 1; }
  echo "data: $target <- $source"
}

restore() {
  # A previous stage's output is the whole working directory, mounted somewhere beneath
  # the path given; locate the repo copy by its run log.
  local prev="${1:-}" log src dir
  [ -n "$prev" ] || return 0
  log=$(find "$prev" -maxdepth 6 -type f -path '*/results/runs.csv' -print -quit)
  [ -n "$log" ] || { echo "no results/runs.csv beneath $prev" >&2; exit 1; }
  src=$(dirname "$(dirname "$log")")
  for dir in results checkpoints runs; do
    if [ -d "$src/$dir" ]; then mkdir -p "$dir"; cp -r "$src/$dir/." "$dir/"; fi
  done
  echo "restored results/, checkpoints/ and runs/ from $src"
}

require_runs() {
  [ -f results/runs.csv ] || {
    echo "results/runs.csv missing: run the earlier stage here, or pass its output" >&2
    exit 1
  }
}

fresh_results() {
  # The clone carries the Mac results. Rows from new hardware must not mix with them:
  # RESULTS.md's noise floor groups identical runs, and hardware moves the numbers.
  [ -f results/runs.csv ] || return 0
  if on_kaggle; then
    rm -f results/*.csv results/*.json results/RESULTS.md results/figures/*.png
    echo "cleared the clone's committed results; this run writes its own"
  elif [ "${FRESH:-0}" = 1 ]; then
    local archive
    archive="results/_previous_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$archive"
    mv results/*.csv results/*.json results/RESULTS.md "$archive"/ 2>/dev/null || true
    echo "moved existing results to $archive"
  else
    echo "results/runs.csv exists and this is not Kaggle. Set FRESH=1 to start over" \
         "(existing results move to results/_previous_*)." >&2
    exit 1
  fi
}

record_env() {
  # Hardware and versions, kept with the results: the README has to say what ran where.
  "$PY" - "$1" <<'PYEOF'
import json, os, platform, sys
from importlib.metadata import PackageNotFoundError, version

import torch

from modeling.tracking import git_dirty, git_sha


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


info = {
    "stage": sys.argv[1],
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "packages": {name: installed(name) for name in (
        "segmentation-models-pytorch", "timm", "albumentations", "numpy",
        "opencv-python", "opencv-python-headless", "onnx", "onnxruntime")},
    "git_sha": git_sha(),
    "git_dirty": git_dirty(),
}
os.makedirs("results", exist_ok=True)
with open(f"results/environment_{sys.argv[1]}.json", "w") as handle:
    json.dump(info, handle, indent=2)
print(json.dumps({k: info[k] for k in ("python", "torch", "cuda", "gpus", "git_sha", "git_dirty")}))
PYEOF
}

gpu_count() { "$PY" -c "import torch; print(torch.cuda.device_count())"; }

run_lane() {
  # run_lane GPU SUITE[:SEED,SEED...] ...   An empty GPU leaves device selection alone.
  local gpu=$1 item suite seeds
  shift
  for item in "$@"; do
    suite=${item%%:*}
    seeds=0
    [ "$item" = "$suite" ] || seeds=${item#*:}
    echo "  [gpu ${gpu:-default}] suite $suite, seeds ${seeds//,/ } -> logs/$suite.log"
    # shellcheck disable=SC2086  # the seed list is meant to split into words
    if [ -n "$gpu" ]; then
      CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_ablations.py --suite "$suite" \
        --seeds ${seeds//,/ } --workers "$WORKERS" > "logs/$suite.log" 2>&1
    else
      "$PY" scripts/run_ablations.py --suite "$suite" \
        --seeds ${seeds//,/ } --workers "$WORKERS" > "logs/$suite.log" 2>&1
    fi || { echo "  suite $suite FAILED; see logs/$suite.log" >&2; return 1; }
  done
}

run_lanes() {
  # run_lanes "SUITES for one lane" "SUITES for another" ...
  # With two or more GPUs, lanes run in parallel, one per GPU; otherwise in sequence.
  local n lane=0 failed=0 list pid
  local -a pids=() items=()
  n=$(gpu_count)
  [ "$n" -gt 0 ] || echo "  no CUDA GPU visible: running on CPU will take days" >&2
  for list in "$@"; do
    read -r -a items <<< "$list"
    if [ "$n" -lt 2 ]; then
      run_lane "" "${items[@]}" || failed=1
    else
      run_lane $((lane % n)) "${items[@]}" &
      pids+=($!)
      lane=$((lane + 1))
    fi
  done
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || failed=1; done
  return "$failed"
}

stage_setup() {
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || { echo "Python >= 3.11 required (the code uses datetime.UTC)" >&2; exit 1; }
  # Keep the image's CUDA build of torch, and its OpenCV if it has one: reinstalling
  # either downloads gigabytes and can break the CUDA pairing.
  local skip='^(torch|torchvision)([<>=!~ ]|$)'
  "$PY" -c 'import cv2' 2>/dev/null \
    && skip='^(torch|torchvision|opencv-python-headless)([<>=!~ ]|$)'
  grep -vE "$skip" requirements.txt > logs/requirements-kaggle.txt
  "$PY" -m pip install -q -r logs/requirements-kaggle.txt
  # Fetch every encoder the suites use now, so a missing internet connection fails here
  # rather than hours into a stage. Needs Internet on (phone-verified account).
  "$PY" - <<'PYEOF'
import segmentation_models_pytorch as smp

for encoder in ("resnet34", "resnet18", "mobilenet_v2"):
    smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", classes=7)
print("ImageNet encoder weights fetched")
PYEOF
  link_data "${1:-}"
  record_env setup
  "$PY" -m pytest tests -q
}

stage_smoke() {
  link_data ""
  "$PY" -m data.prepare_masks --split-mode official \
    --stats-out logs/smoke_dataset_stats.json > logs/smoke_prepare.log 2>&1
  local monitor=""
  if command -v nvidia-smi > /dev/null; then
    nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 2 \
      > logs/smoke_gpu_util.log & monitor=$!
  fi
  "$PY" - <<'PYEOF'
import time

from modeling.train import train

start = time.time()
result = train("data/processed/level1_official", epochs=2, run_name="smoke",
               out_dir="/tmp/roadcred_smoke", results_csv="/tmp/roadcred_smoke/runs.csv",
               tensorboard=False)
per_epoch = (time.time() - start) / 2
# Reference: ~62 s/epoch for this recipe on the M1 Pro behind the reported runs
# (loss_ce_s0: 1,870 s for 30 epochs, validation included). The first epoch here also
# pays for cuDNN start-up, so these figures run slightly pessimistic.
print(f"\n{result['device']}: {per_epoch:.0f} s/epoch, about {62 / per_epoch:.1f}x the M1 Pro")
print(f"projected training: ~{13 * per_epoch / 62:.1f} h on one GPU, "
      f"~{7.5 * per_epoch / 62:.1f} h with two")
PYEOF
  if [ -n "$monitor" ]; then
    kill "$monitor" 2> /dev/null || true
    awk '{s += $1; n++} END {if (n) printf "mean GPU utilisation: %.0f%%%s\n", s / n, \
      (s / n < 80 ? " - below ~80%, so try WORKERS=2" : "")}' logs/smoke_gpu_util.log
  fi
}

stage_train1() {
  fresh_results
  link_data ""
  record_env train1
  "$PY" -m data.prepare_masks --split-mode all --target level1 > logs/prepare_masks.log 2>&1
  # Balanced from M1 Pro times: split + loss ~4.1 h, seedvar ~3.0 h.
  run_lanes "split loss" "seedvar:0,1,2"
}

stage_train2() {
  restore "${1:-}"
  require_runs
  link_data ""
  record_env train2
  "$PY" -m data.prepare_masks --split-mode all --target level1 > logs/prepare_masks.log 2>&1
  # Balanced from M1 Pro times: arch + heldout ~2.2 h, stride + pretrain ~2.5 h.
  run_lanes "arch heldout" "stride pretrain"
  for split in val test; do
    "$PY" -m evaluation.eval --checkpoint checkpoints/heldout_sequence3_s0.pt \
      --data data/processed/level1_sequence3 --split "$split" \
      --out "results/heldout_${split}.json" > "logs/heldout_${split}.log" 2>&1
  done
}

stage_analyse() {
  restore "${1:-}"
  require_runs
  link_data ""
  record_env analyse
  "$PY" -m data.prepare_masks --split-mode all --target level1 > logs/prepare_masks.log 2>&1
  PY="$PY" ./scripts/finalize.sh 2>&1 | tee logs/finalize.log
  echo "Latency here is a shared Kaggle CPU's. For deployment figures, re-run"
  echo "compression.quantize and compression.pareto on the machine that will serve."
}

main() {
  cd "$(dirname "$0")/.."
  export PYTHONPATH=.
  mkdir -p logs
  local stage=${1:-}
  shift || true
  case "$stage" in
    setup) stage_setup "$@" ;;
    smoke) stage_smoke ;;
    train1) stage_train1 ;;
    train2) stage_train2 "$@" ;;
    analyse) stage_analyse "$@" ;;
    *) sed -n '2,8p' "$0" >&2; exit 2 ;;
  esac
  if [ "$stage" != smoke ]; then
    echo "stage $stage done; outputs in results/, checkpoints/ and logs/"
  fi
}

# Sourcing (from bash) defines the functions without running a stage, so they can be
# tested; outside bash BASH_SOURCE is unset and nothing runs.
if [ "${BASH_SOURCE[0]-}" = "$0" ]; then
  main "$@"
fi
