# RoadCred — Unstructured Road Scene Segmentation

![A dirt road in India with two parked vehicles and pedestrians, shown three ways: the raw input frame, the same frame with seven predicted classes painted over it, and a per-pixel confidence map that is bright inside regions and dark along every class boundary](docs/images/demo_hero.png)

**Left to right: what the camera sees, what the model says, and where the model is unsure.**

The first two panels are the task — 7 classes over an unstructured Indian road, where there
is no lane marking, the drivable surface is dirt, and the boundary between road and
not-road is a judgement call rather than a painted line. **The third panel is the
project.** It is the model's own calibrated
confidence, and it renders as an *edge map*: bright and certain across road, sky and
vegetation, dark along every boundary between them. The model is least sure exactly where
the classes meet — which turns out to explain its per-class accuracy far better than class
rarity does, and is the finding the rest of this README builds to.

---

7-class semantic segmentation of Indian road scenes on the India Driving Dataset, built as
a study in *evaluation rigour* rather than leaderboard chasing. A hand-written PyTorch
training loop drives every experiment; a controlled ablation varies one factor at a time;
and the headline result is a measurement of how badly a naive train/test split flatters a
model on this data. Ships with a calibrated FastAPI inference service and a
dependency-free demo page.

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
> shipped the model as an INT8-quantized ONNX service behind FastAPI, which reduces each
> frame to a bounded drivable-path score that flags its own known failure mode.
> 144 tests in CI, including an assertion that the generalization split cannot leak.

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
onto **drivable**. That is precisely the boundary this project measures. `detect_level` now
recognises IDD Lite's masks as level-1, and data preparation audits every mask set before
writing anything. No single mask can reveal this bug — 0–6 is a legal level-3 range — so the
audit works across the whole dataset: masks read as a finer level whose IDs never leave the
level-1 range raise. Per mask, an ID beyond the assumed level's range raises too, which
catches the reverse mistake.

### 3. Make the evaluation honest before optimising the number

IDD groups frames into drive sequences. Frames within a drive are seconds apart on the same
road in the same light. Split those at random and validation frames have near-duplicates in
training.

Measured, not assumed: a random frame split puts **94.6% of validation frames in a drive
that also appears in training**. `tests/test_sequence_split.py` asserts that the honest
splits cannot leak, so the claim is enforced by CI rather than merely documented.

All three splits are built and trained, **three seeds per arm**, so the inflation is a
distribution rather than a point estimate:

| split | seeds | mIoU | std |
|---|---|---|---|
| drive-disjoint | 3 | 0.6768 | 0.0032 |
| random frame — **leaky control** | 3 | 0.6894 | 0.0009 |

**Inflation +0.0126 mIoU (Welch's t-test, p = 0.016).** The effect clears its own noise
floor, which a single-seed measurement could never have shown.

**The magnitude is smaller than I expected, and the reason is the interesting part.** 94.6%
contamination moved mIoU by only ~1.3 points because frames within an IDD drive sit a
median of ~4,200 frame indices apart — they share scene and lighting without being
near-duplicates. *Contaminated is not the same as duplicated.* The same fact independently
killed the temporal-consistency metric (§6). A dataset of genuinely consecutive frames
would be expected to show a far larger gap; this one does not, and the honest thing is to
report the modest number rather than the dramatic one I went looking for.

The leaky arm also had the smaller seed spread (std 0.0009 against 0.0032). With three seeds
per arm that difference is not distinguishable from chance (F-test, p = 0.13), and every
seed trained on the same split, so it cannot speak to which drives land where. It is
recorded as an observation, not a finding.

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
way calibration results get overstated. The pixels are drawn uniformly from every
validation frame; an earlier version filled its budget from the first 56 frames in filename
order and fitted T = 1.240, where the uniform sample gives 1.304. `tests/test_calibration.py`
checks that a known temperature of 3.0 is recovered.

### 6. Measure robustness without inventing a dataset

No adverse-weather dataset is used. Ten corruptions (fog, rain, low light, motion blur,
noise, JPEG …) are implemented and applied **at test time only** — the training augmentation pipeline
deliberately excludes them — so the degradation measures genuine distribution shift rather
than a train/test augmentation mismatch. Each corrupted set is re-evaluated after
**test-time BatchNorm adaptation**, which re-estimates running statistics on the shifted
data with no labels and no gradient steps. All ten are evaluated at severities 1 and 3, and
five of them (fog, rain, low light, motion blur, Gaussian noise) also test calibration under
shift; severities 2/4/5 are implemented and driven by the same command.

**One idea was cut after checking the data.** Frame-to-frame temporal flicker would be the
natural stability metric, but IDD Lite's frames are not temporally adjacent — the median gap
between consecutive frames of a drive is ~4,200 frame indices, and only 0.4% are within 30
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

### 8. Reduce it to the number someone will actually ask for — and bound that number

A per-pixel class map is not what a consumer of this model wants. The question underneath
is always some version of *can the vehicle go forward*, and if the project declines to
answer it, a reader answers it anyway by eyeballing the overlay — less accurately, and with
no caveats attached. So the serving layer computes one explicitly: a **drivable-path
score** over the forward corridor, a trapezoid ahead of the camera, wide near the vehicle
and narrowing with distance.

```
score = coverage × mean_confidence × (1 − obstruction)
```

**A product, not a weighted sum.** The three terms are not interchangeable, and an average
would call a corridor that is 100% road but blocked by a truck "mostly fine". Any one
component collapsing should collapse the score, and under a product it does.

**Every component is returned alongside the score**, because the aggregate is ambiguous in
exactly the way that matters: a corridor scoring low because the model is unsure and one
scoring low because a bus is parked in it are different situations that a single number
cannot separate.

**It flags its own known failure mode.** This project measured that about 20% of true
`non-drivable` is absorbed into `drivable`, and of the interventions tried, only class
re-weighting improved that beyond run-to-run noise — so wherever absorption lands inside the
corridor, the score is optimistic. Whenever the model draws a road edge in the corridor or
within 1% of image width of it, the response sets `road_edge_caveat` and the demo page
renders the warning rather than burying it. On validation the flag fired on all 22 frames
where absorption reached inside the corridor, and on half of all frames overall: it trades
false alarms for never staying silent. The rule it replaced — fire when the corridor holds
*predicted* non-drivable — caught 14 of the 22, and was blind by construction to total
absorption ([`RESULTS.md`](results/RESULTS.md) §8d).

It is a **description of the segmentation, not an assessment of the road**, and calling it
"drivability" does not make it one. The out-of-scope restriction in
[`MODEL_CARD.md`](MODEL_CARD.md) is unchanged: not for vehicle control, driver assistance,
or navigation.

---

## Results

See **[`results/RESULTS.md`](results/RESULTS.md)** — generated from the CSVs in `results/`
by `scripts/build_results.py`, so no number in it is hand-transcribed.

Headline figures measured so far:

| result | value |
|---|---|
| Best validation mIoU | **0.6979** (DeepLabV3+/ResNet-34, encoder stride 8) |
| Random-split leakage, 3 seeds/arm | **+0.0126 mIoU** inflation (Welch's t-test p = 0.016) |
| Held-out test set, one run | val 0.6669 → test 0.6690: **no measurable early-stopping optimism** |
| Encoder stride 16 → 8 | **+0.0077 mIoU** (3.5× the identical-run range), concentrated in boundary-heavy classes |
| Run-to-run noise, identical config *and seed* | **0.0022 mIoU** across 4 runs; up to **0.0145** per class |
| INT8 quantization | **4.4× faster, 3.9× smaller, −0.0001 mIoU** (42.4 ms, 22.9 MB, CPU) |
| Distillation vs same student trained alone | **+0.0010 mIoU**, inside run-to-run noise: unresolved at one seed |
| Calibration | overconfident at **T = 1.304**; ECE 0.0272 → **0.0039** (−86%) |
| Confidence under severe motion blur | ECE **0.0041 → 0.0600** — 15× worse, accuracy only 0.89 → 0.82 |
| Test-time BN adaptation, severe shot noise | 0.2688 → **0.5730** (+0.304, no labels, no gradients) |
| FGSM, ε = 4/255 | **71.7%** of clean mIoU |

### It is not rarity. It is geometry.

The obvious reading of the per-class table is that errors track class rarity. That reading
is a confound, and this project ended up disproving its own earlier claim.

What predicts a class's IoU is how much of it is *boundary*, not how rare it is. The
decisive case: `barrier-structures` is **8.6× more common** than `living-thing` and scores
about the same IoU. Rarity cannot explain that; thinness predicts it.

Correlation was only the hypothesis — n = 7 classes, with the two predictors themselves
collinear at r = −0.91. So it was tested directly. Halving the encoder's output stride
doubles the resolution of the features the decoder sees and changes nothing else:

| class | pixels near a border | Δ IoU at stride 8 |
|---|---|---|
| living-thing | 55% | **+0.018** |
| non-drivable | 40% | **−0.017** |
| barrier-structures | 35% | **+0.026** |
| vehicles | 22% | +0.014 |
| construction-vegetation | 19% | +0.010 |
| sky | 12% | +0.001 |
| drivable | 7% | +0.001 |

The two flattest classes gain +0.001; the boundary-heavy ones gain 10–26× that, and every
change except `drivable`'s falls outside the range four identical runs span. **That is a
controlled intervention, not a correlation.**

And `non-drivable` goes the wrong way. Per-class IoU is far noisier than mIoU — four runs
with identical config *and* seed span 0.011 on `non-drivable` and 0.015 on `living-thing` —
so each single-seed intervention is read against that range ([`RESULTS.md`](results/RESULTS.md)
§1c, §3b). **`living-thing` improves beyond it under all six sharpening interventions**
(Dice, Tversky, boundary-weighted loss, U-Net, FPN, stride 8; boundary-weighted loss only
marginally). **`non-drivable` drops beyond it under three** — U-Net, stride 8 and, marginally,
boundary-weighted loss — stays inside it under Dice, Tversky and FPN, and improves beyond it
only under explicit class re-weighting. That still splits one apparent failure mode into
two: `living-thing` is a small compact object that wants resolution, while `non-drivable` is
a thin strip abutting `drivable` (32% of all pixels) that gets absorbed by it, and the two
interventions that add the most effective resolution make that absorption worse. The
evidence for that second half is three interventions, not the six an earlier version of
this README claimed.

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

# 4. Train one model with the recipe behind every reported run (~30 min on an M1 Pro)
PYTHONPATH=. .venv/bin/python -m modeling.train --data data/processed/level1_official \
    --epochs 30 --patience 8

# 5. Or reproduce every table and figure (~14 h)
./scripts/reproduce.sh
```

On Kaggle's free GPUs, [`scripts/kaggle_stage.sh`](scripts/kaggle_stage.sh) runs the same
pipeline in stages that each fit one session, spread across both GPUs; its header has the
notebook cells.

### Serving and the demo

```bash
# Export + quantize, then serve
PYTHONPATH=. .venv/bin/python -m compression.quantize --checkpoint checkpoints/<best>.pt
PYTHONPATH=. .venv/bin/uvicorn serving.api:app --reload --port 8000
```

Open <http://localhost:8000>. Drop in a road-scene image and toggle the overlay / mask /
confidence layers.

![Demo page: the uploaded frame beside the predicted classes blended over it, with the drivable-path score, calibrated confidence and per-class composition in the right-hand column](docs/images/demo_overlay.png)

The right-hand column is where the reasoning is. **Drivable path** (§8) scores the forward
corridor and shows the components that produced it — on this frame, 81.3% of the corridor
predicted drivable at 97.8% mean confidence, against 18.7% blocked by the two vehicles
ahead. That 18.7% is the whole reason the score sits at 0.647 rather than 0.80, and the
breakdown is what makes the difference readable. The road-edge caveat stays down: the model
draws no road edge in or near this corridor, and ground truth puts none there either. **Prediction** carries the calibrated mean
confidence and the temperature it was scaled by; **Class composition** gives per-class
pixel share and confidence.

The **mask** layer drops the photograph entirely and shows the raw class map:

![Demo page on the mask layer: the predicted class map alone, flat colour per class, with no input image blended underneath](docs/images/demo_mask.png)

Worth a look because it is unflattering by design: with the photograph gone there is
nothing left to lend the output plausibility, and shape errors stand on their own. The
vehicles hold clean silhouettes. The class table shows `non-drivable` at 0.0% of the frame
— a dirt road with an unpaved shoulder, and almost none of it called non-drivable.

An earlier version of this README called that the absorption failure mode of §8 caught in
the act. Checked against ground truth, it is not: IDD's annotators labelled this shoulder
drivable as well, putting only 0.3% of the frame in `non-drivable`, and the model's 37.3%
drivable matches their 37.3%. On this road the edge is a judgement call, and the model
makes the annotators' call. The absorption is real — about a fifth of true `non-drivable`
across the validation set ([`RESULTS.md`](results/RESULTS.md) §9, §8d) — but this frame does
not show it, which is why the claim rests on ground truth rather than on a screenshot.

**The confidence view is the one worth looking at**, because it reproduces this project's
per-class finding on a single frame, with no analysis:

![Per-pixel calibrated confidence, rendering as an edge map: boundaries dark, region interiors bright](docs/images/demo_confidence.png)

It renders as an **edge map**. Region interiors — road, sky, vegetation — are uniformly
bright; every class boundary is dark. The model is least certain exactly where classes
meet, which is the same effect the per-class table measures at r = −0.91 against boundary
share. Read the class panel alongside it and the ordering repeats: on this frame the flat
classes score 89–95% mean confidence while the thin ones fall to 40–66%.

That is the whole argument of *It is not rarity. It is geometry.* above, visible without
reading a number.

The demo is a **single dependency-free HTML file** served by the API itself — no npm, no
build step, no bundle that can drift out of sync with the endpoint it calls. It replaced a
React/Vite/TypeScript frontend whose two dashboard pages re-rendered the same CSVs that
[`results/RESULTS.md`](results/RESULTS.md) already generates, and generates better. The
one page that did something the report could not — run the model — needed no framework.

### Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v      # 144 tests, no dataset required
```

Tests needing the real IDD tree skip themselves, so CI runs green without a download.

---

## Repository layout

```
data/         label mapping (name-keyed), frame discovery, split strategies, dataset build
modeling/     hand-written training loop, losses, dataset, model factory, run tracking
evaluation/   metrics, boundary IoU, calibration, corruptions, stability, error analysis
compression/  ONNX export with parity check, INT8 PTQ, latency benchmarking
serving/      FastAPI service (ONNX Runtime inference), drivable-path scoring
serving/static/index.html   the demo page, dependency-free
docs/images/  demo screenshots, regenerated by scripts/capture_demo.py
scripts/      ablation drivers, reproduction, RESULTS.md generator, Kaggle stage runner
tests/        144 tests incl. the anti-leakage assertion
```

## Limitations

- **Scale.** 1607 labelled frames at 320×227. Absolute mIoU is not comparable to results on
  full IDD 20K or Cityscapes, and is not claimed to be.
- **Corruptions are synthetic.** The fog model is depth-independent because IDD Lite ships
  no depth map; real adverse-weather data would be a stronger test.
- **One held-out test run.** IDD Lite withholds its test labels, so validation does double
  duty — early-stopping criterion and reporting set — everywhere except one experiment
  (`RESULTS.md` §5b), where a drive-disjoint test partition was carved from the labelled
  frames. That measures early-stopping optimism for one run. It does not measure the
  optimism of picking the best of many configurations on validation, which the headline
  0.6979 carries. The calibration split (§5) is clean *within* validation — temperature is
  fitted and evaluated on disjoint pixel halves — but inherits this. Read every other
  figure as a validation number.
- **Seed variance is measured for the headline, assumed elsewhere.** The split experiment
  was run across three seeds per arm (drive-disjoint 0.6768 ± 0.0032, random-frame
  0.6894 ± 0.0009; inflation +0.0126, p = 0.016). Every *other* ablation cell is one seed.
  Four runs with identical config **and identical seed** span 0.0022 mIoU — MPS offers no
  determinism guarantee and `cudnn.deterministic` is CUDA-only — and their per-class IoU
  spans up to 0.0145. Single-seed mIoU differences under ~0.003, and per-class differences
  inside the ranges in `RESULTS.md` §1c, are unresolved.
- **Most recorded runs have `git_dirty=True`** (`RESULTS.md`'s header gives the count). The
  run log stores a commit SHA per row, but where the tree was dirty that SHA does not fully
  identify the code that ran.
- **The road-edge caveat's margin was chosen on validation**, the same frames its recall is
  reported on, so that recall is optimistic; the margin sweep is published beside it.
- **`train_seconds` is wall-clock, not compute.** Runs left going overnight include machine
  sleep, so that column is not a benchmark. Latency figures in the compression tables are
  measured properly, under pinned threads.
- **Architecture comparison is decoder-only.** All arms share an encoder family and the same
  training recipe, so it does not speak to architectures with different training needs.

## Corrections

An audit of this README against the code, the run log and the data (November 2026) found
claims that did not hold. The text above is corrected; this is what changed.

- **Held-out test.** Its run had trained from random initialisation: the training command
  turned an omitted `--encoder-weights` into none. Retrained with ImageNet weights it scores
  val 0.6669 → test 0.6690, not 0.5782 → 0.5778. The accidental run is kept as a
  pretraining comparison (+0.09 test mIoU for ImageNet).
- **Distillation.** The standalone student trained at batch size 16, the distilled one at 8.
  On one shared recipe the gain is +0.0010, not +0.0028, and inside run-to-run noise.
- **Calibration.** Its pixels came from the first 56 of 204 validation frames. Sampled
  uniformly: T = 1.304, not 1.240; severe motion blur is 15× the clean ECE, not 25×.
- **Noise and per-class claims.** Four identical runs span 0.0022 mIoU (stated: 0.0012) and
  up to 0.0145 per class. Against that, three interventions degrade `non-drivable`, not six,
  and the leaky split's smaller seed spread is not significant.
- **Road-edge caveat.** It could not fire when absorption was total; it now keys on the road
  edge the model draws, and its recall is measured.
- **Demo narrative.** The mask-layer screenshot was presented as absorption caught in the
  act. Its ground truth has only 0.3% `non-drivable`, and the model agrees with it.
- **Smaller.** Boundary IoU counted the image edge as boundary; the label-level range check
  never ran in the dataset build; some report figures were hardcoded rather than read from
  result files. Each fix has a regression test.
