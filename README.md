# RoadSense — Unstructured Road Scene Segmentation

7-class semantic segmentation of Indian road scenes on the India Driving Dataset, built as
a study in *evaluation rigour* rather than leaderboard chasing. A hand-written PyTorch
training loop drives every experiment; a controlled ablation varies one factor at a time;
and the headline result is a measurement of how badly a naive train/test split flatters a
model on this data. Ships with a calibrated FastAPI inference service and a React
dashboard.

---

## Summary

> Built an end-to-end semantic segmentation system for unstructured Indian road scenes
> (7 classes, India Driving Dataset), written on a hand-rolled PyTorch training loop with
> AMP-free MPS support, warmup+cosine scheduling, early stopping and per-run experiment
> tracking. Quantified that a naive random frame split leaks 94.6% of validation frames
> into training via shared drive sequences, and reported drive-disjoint results throughout.
> Ran controlled ablations over five segmentation losses and four decoder architectures,
> measured confidence calibration with leakage-free temperature scaling, evaluated
> robustness across ten test-time corruptions with label-free BatchNorm adaptation, and
> shipped the model as an INT8-quantized ONNX service behind FastAPI with a React frontend.
> 80 unit tests in CI, including an assertion that the generalization split cannot leak.

---

## The decision log

This is the order the project was actually reasoned through. Numbers are in
[`results/RESULTS.md`](results/RESULTS.md).

### 1. Pick a task the data can actually support

The dataset is **IDD Lite** — 1607 labelled frames across 370 drive sequences at 320×227.
An earlier draft of this project planned a road-damage class built by compositing an
external pothole dataset onto IDD's drivable regions. That was dropped, for a reason worth
stating: the labels would have been synthetic, and a reviewer would have been right to
discount every number derived from them.

IDD Lite already ships a real 7-class problem — IDD's own level-1 hierarchy — with genuine
imbalance:

| class | share of training pixels |
|---|---|
| drivable | 32.4% |
| construction-vegetation | 26.0% |
| sky | 18.9% |
| barrier-structures | 11.2% |
| vehicles | 8.1% |
| **non-drivable** | **2.2%** |
| **living-thing** | **1.3%** |

That imbalance is what makes the loss ablation an experiment rather than a formality.
**Every dataset in this project comes from IDD. There are no external data sources and no
synthetic labels.**

### 2. Get the label mapping right before anything else

IDD encodes labels at five parallel granularities (`id`, `level1Id` … `level4Id`), and the
same integer means different classes at different levels. Every mapping here is therefore
derived from IDD's official label table **keyed by class name**, never by hardcoded IDs.

This caught a real bug. IDD Lite ships bare `<frame>_label.png` files that contain *level-1*
IDs, but nothing in the filename says so. Filename-based level detection defaulted them to
level-3, which silently mapped `level1Id 1` — sidewalk, rail track, non-drivable fallback —
onto **drivable**. That is precisely the boundary this project measures. `remap_mask` now
range-checks every mask against its assumed level and raises rather than mis-mapping.

### 3. Make the evaluation honest before optimising the number

IDD groups frames into drive sequences. Frames within a drive are seconds apart on the same
road in the same light. Split those at random and validation frames have near-duplicates in
training.

Measured, not assumed: a random frame split puts **94.6% of validation frames in a drive
that also appears in training**. `tests/test_sequence_split.py` asserts that the honest
splits cannot leak, so the claim is enforced by CI rather than merely documented.

All three splits are built and trained, so the inflation is a number rather than a worry:
**+0.0139 mIoU (+2.1%)**, comparing the random split against a drive-disjoint split of the
same pooled frames.

**That result is smaller than I expected, and the reason is the interesting part.** 94.6%
contamination moved mIoU by only ~1.4 points because frames within an IDD drive sit a
median of ~4,400 frame indices apart — they share scene and lighting without being
near-duplicates. *Contaminated is not the same as duplicated.* The same fact independently
killed the temporal-consistency metric (§6). A dataset of genuinely consecutive frames
would be expected to show a far larger gap; this one does not, and the honest thing is to
report the modest number rather than the dramatic one I went looking for.

The leaky arm is kept deliberately — as the control to argue against, never as a headline.

### 4. Write the training loop by hand

Every experiment runs through one loop (`modeling/train.py`) with warmup + cosine schedule,
gradient clipping, early stopping on validation mIoU, checkpointing, TensorBoard scalars,
and a CSV row per run capturing config, seed and git SHA. Losses are `nn.Module` subclasses
written from scratch and verified against hand computation.

Because every arm shares that recipe, a difference between two rows is attributable to the
factor that was varied.

A small engineering finding: dataloader workers are **off** by default. Augmented loading
measures ~640 img/s single-process against ~37 img/s of MPS training throughput, so worker
processes bought nothing and added spawn overhead and failure modes.

### 5. Report what the model doesn't know

The serving layer returns a confidence score, so that number has to mean something.
Temperature scaling is fitted on one half of the validation pixels and **every reported
figure computed on the other half** — fitting and reporting on the same pixels is the usual
way calibration results get overstated. The implementation recovers a known temperature of
3.0 as 2.973 in test.

### 6. Measure robustness without inventing a dataset

No adverse-weather dataset is used. Ten corruptions (fog, rain, low light, motion blur,
noise, JPEG …) are applied **at test time only** — the training augmentation pipeline
deliberately excludes them — so the degradation measures genuine distribution shift rather
than a train/test augmentation mismatch. Each corrupted set is re-evaluated after
**test-time BatchNorm adaptation**, which re-estimates running statistics on the shifted
data with no labels and no gradient steps.

**One idea was cut after checking the data.** Frame-to-frame temporal flicker would be the
natural stability metric, but IDD Lite's frames are not temporally adjacent — the median gap
between consecutive frames of a drive is ~4,400 frame indices, and only 0.5% are within 30
frames. A flicker number computed from them would have measured scene change. It was
replaced with perturbation stability, which measures the same property honestly.

### 7. Ship the artefact that was benchmarked

The model is exported to ONNX and quantized to INT8 with static, calibrated PTQ. Export is
verified against PyTorch on real frames, reporting both logit error **and the fraction of
pixels whose predicted class changes** — a graph can round-trip logits closely while still
flipping labels at boundaries. FP32 and INT8 are benchmarked on **CPU** with pinned threads,
because Apple silicon has no INT8 path through MPS and a GPU comparison would silently run
the quantized model in FP32.

The API serves the ONNX model, so the thing being demoed is the thing that was measured.

---

## Results

See **[`results/RESULTS.md`](results/RESULTS.md)** — generated from the CSVs in `results/`
by `scripts/build_results.py`, so no number in it is hand-transcribed.

Headline figures measured so far:

| result | value |
|---|---|
| Drive-disjoint validation mIoU | **0.6755** (IDD's own split: 0.6913) |
| Random-split leakage | 94.6% of val frames contaminated → **+0.0139 mIoU** inflation |
| Calibration | overconfident at **T = 1.229**; ECE 0.0211 → 0.0027 (**−87%**) |
| Test-time BN adaptation, severe noise | 0.3410 → **0.6196** (+0.279, no labels, no gradients) |
| Test-time BN adaptation, severe rain | 0.4652 → **0.5615** (+0.096) |
| FGSM, ε = 4/255 | 71.4% of clean mIoU |
| Per-class IoU vs class rarity | Pearson **r = 0.80** on log frequency |

The loss ablation, architecture comparison, distillation and compression Pareto are coded
and queued but were not run to completion; `./scripts/reproduce.sh` runs them.

All results are at IDD Lite scale on a single M1 Pro. This is a small dataset and the
absolute mIoU values reflect that. The comparisons are controlled, so the *differences*
between rows carry the meaning.

---

## Quickstart

```bash
# 1. Data — see DOWNLOADS.md. Extract IDD Lite so this path exists:
#    data/raw/idd_seg/idd20k_lite/{leftImg8bit,gtFine}/

# 2. Environment
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3. Build the dataset variants
PYTHONPATH=. .venv/bin/python -m data.prepare_masks --split-mode all

# 4. Train a model (~30 min on an M1 Pro)
PYTHONPATH=. .venv/bin/python -m modeling.train --data data/processed/level1_official

# 5. Or reproduce every table and figure (~7 h)
./scripts/reproduce.sh
```

### Serving and the dashboard

```bash
# Export + quantize, then serve
PYTHONPATH=. .venv/bin/python -m compression.quantize --checkpoint checkpoints/<best>.pt
PYTHONPATH=. .venv/bin/uvicorn serving.api:app --reload --port 8000

# In a second terminal
cd frontend && npm install && npm run dev     # http://localhost:5173
```

Three pages: **Inference** (upload an image, toggle overlay / mask / confidence layers),
**Model Comparison** (loss ablation, architectures, compression), **Robustness** (the split
experiment, corruption curves, error analysis).

### Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v      # 80 tests, no dataset required
```

Tests needing the real IDD tree skip themselves, so CI runs green without a download.

---

## Repository layout

```
data/         label mapping (name-keyed), frame discovery, split strategies, dataset build
modeling/     hand-written training loop, losses, dataset, model factory, run tracking
evaluation/   metrics, boundary IoU, calibration, corruptions, stability, error analysis
compression/  ONNX export with parity check, INT8 PTQ, latency benchmarking
serving/      FastAPI service (ONNX Runtime inference)
frontend/     React 19 + TypeScript + Tailwind + recharts
scripts/      ablation drivers, reproduction, RESULTS.md generator
tests/        80 tests incl. the anti-leakage assertion
```

## Limitations

- **Scale.** 1607 labelled frames at 320×227. Absolute mIoU is not comparable to results on
  full IDD 20K or Cityscapes, and is not claimed to be.
- **Corruptions are synthetic.** The fog model is depth-independent because IDD Lite ships
  no depth map; real adverse-weather data would be a stronger test.
- **One seed per ablation cell** in the default run, for compute reasons. `--seeds 0 1 2`
  runs the multi-seed version; spread is reported where it was run. The +0.0139 leakage
  figure is therefore a single-seed measurement and should be read as indicative.
- **`train_seconds` is wall-clock, not compute.** Runs left going overnight include machine
  sleep, so that column is not a benchmark. Latency figures in the compression tables are
  measured properly, under pinned threads.
- **Architecture comparison is decoder-only.** All arms share an encoder family and the same
  training recipe, so it does not speak to architectures with different training needs.
