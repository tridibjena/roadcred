# Model Card — RoadCred 7-class road scene segmentation

## Model details

- **Task:** semantic segmentation, 7 classes, single label per pixel.
- **Classes:** `drivable`, `non-drivable`, `living-thing`, `vehicles`,
  `barrier-structures`, `construction-vegetation`, `sky` — IDD's own level-1 hierarchy,
  unmodified.
- **Architecture:** DeepLabV3+ decoder over a ResNet-34 encoder
  (`segmentation-models-pytorch`), ImageNet-initialised encoder, **encoder output stride
  8**. Stride 8 rather than the default 16 is the single change that produced the shipped
  checkpoint: it doubles the resolution of the features the decoder sees, costs ~2x
  training time, and buys +0.0077 mIoU concentrated almost entirely in the thin classes
  (see *Known failure modes*). 22.4M parameters.
- **Input:** RGB, resized to 320×224, ImageNet normalisation.
- **Output:** per-pixel logits; confidence is the temperature-scaled softmax maximum.
- **Serving artefact:** ONNX (opset 18), INT8-quantized with static calibrated PTQ, run
  through ONNX Runtime on CPU. **22.9 MB and 42.4 ms/frame, against 90.0 MB and 185.6 ms
  for FP32 — 3.9x smaller and 4.4x faster for a 0.0001 mIoU loss.** Both precisions
  benchmarked on CPU with pinned threads; Apple silicon has no INT8 path through MPS, so
  a GPU comparison would silently run the quantized graph in FP32.
- **Training:** hand-written PyTorch loop — AdamW, linear warmup then cosine decay,
  gradient clipping at 1.0, early stopping on validation mIoU.

## Intended use

Built as a **portfolio and research artefact** demonstrating segmentation-model evaluation
methodology. Appropriate uses: studying evaluation protocol, class imbalance, calibration,
robustness under distribution shift, and CPU deployment of a segmentation model.

## Out-of-scope use

**This model must not be used for vehicle control, driver assistance, navigation, or any
safety-relevant decision.** It is trained on 1403 low-resolution frames, has an IoU of well
under 0.5 on its rarest classes, and has never been validated against any safety standard.
The `living-thing` class in particular — the one whose errors would matter most — is the
weakest, at roughly 1.3% of training pixels.

It is also not validated outside its training distribution: daytime Indian road scenes from
IDD. Behaviour on other geographies, camera mountings, sensors, or night footage is unknown.

**The drivable-path score is covered by this restriction, not exempt from it.** The serving
layer reduces each frame to a single number over a forward corridor
(`score = coverage × mean_confidence × (1 − obstruction)`; see `serving/drivability.py`).
That number is a **descriptive statistic computed from the model's own output** — how much
of the corridor the model called drivable, how confident it was, and how much of it
something else occupies. It describes the segmentation, not the road, and naming it
"drivability" does not turn it into a safety assessment. Two properties are deliberate:
every component is returned alongside the score, because a corridor scoring low from
uncertainty and one scoring low from an obstruction are different situations a single
number cannot distinguish; and the response carries a `road_edge_caveat` flag whenever the
corridor spans a drivable/non-drivable border, because this model absorbs 17–20% of true
`non-drivable` into `drivable` and the score is therefore **optimistic at exactly the
boundary that matters most** (see *Known failure modes*).

## Training data

**India Driving Dataset — IDD Lite (`idd20k_lite`)**, from IIIT Hyderabad
(https://idd.insaan.iiit.ac.in/). 1607 labelled frames across 370 drive sequences at
320×227; 1403 train / 204 validation under IDD's own split.

Reported validation mIoU is **0.6979** on IDD's own drive-disjoint split. A separate
three-way split — train / val / test, all mutually drive-disjoint — was built to obtain a
number model selection never influenced; see *Evaluation protocol*.

Every dataset used is from IDD. There are no external data sources and no synthetic labels.

Class balance is strongly skewed — `drivable` 32.4% of pixels against `living-thing` 1.3%.
Per-class metrics are reported alongside mIoU for this reason; a headline mIoU alone hides
which classes fail.

## Evaluation protocol

- **Splits are drive-disjoint.** IDD's own train/val split shares no drive sequence
  (309 vs 61 drives, zero overlap). A random frame-level split would place 94.6% of
  validation frames in a drive that also appears in training; that arm is reported only as
  a control, never as a headline.
- **Metrics** come from a single confusion matrix accumulated over the whole split. Classes
  absent from the ground truth are `NaN`, not zero. Boundary IoU is reported alongside
  region IoU.
- **A genuinely held-out test set exists.** IDD withholds its own test labels, so a third
  partition was carved from the labelled frames, drive-disjoint from *both* train and
  validation — a test set sharing drives with validation is contaminated through model
  selection just as surely as through training. On that split: **val 0.5782, test 0.5778,
  selection optimism +0.0004.** Early stopping bought itself essentially nothing, which is
  the evidence that the validation figures elsewhere are honest generalization estimates.
  (Absolute mIoU is lower there because holding out two partitions leaves ~940 training
  frames instead of 1403.)
- **Calibration** is fitted on one half of the validation pixels and reported on the other.
- **Robustness** uses test-time corruptions that never appear in training augmentation, and
  is additionally measured *for the confidence score*, not only for accuracy.
- **Run-to-run noise is measured, not assumed.** Two runs with identical configuration and
  identical seed differ by **0.0012 mIoU** — MPS provides no determinism guarantee, and
  `torch.backends.cudnn.deterministic` is CUDA-only. Any difference below roughly that
  magnitude is not interpretable. Seed variance is larger again: **±0.0032** across three
  seeds on the drive-disjoint split.
- **Reproducibility:** each run logs its config, seed and git SHA to `results/runs.csv`.

## Limitations

- Small dataset and low resolution; absolute mIoU is not comparable to full IDD 20K or
  Cityscapes results, and no such comparison is claimed.
- Corruptions are synthetic. The fog model is depth-independent because IDD Lite ships no
  depth map, so it under-represents real fog, which attenuates with distance.
- Most ablation cells are one seed, for compute reasons; the split experiment is the
  exception and was run across three seeds. Differences smaller than ~0.003 mIoU elsewhere
  in this project should be read as unresolved rather than real.
- The architecture comparison varies the decoder within one training recipe, so it says
  nothing about architectures needing different recipes.
- Ground truth is treated as correct. No annotation-noise audit was performed.

## Known failure modes

Per-class IoU on this model does not track class *rarity* so much as class *geometry*. The
share of a class's pixels lying within 2px of a class boundary predicts its IoU better than
its frequency does, and the decisive case is `barrier-structures`: 8.5x more common than
`living-thing` and scoring about the same. Halving the encoder stride — changing nothing
else — improved the boundary-heavy classes by +0.018 to +0.026 and the two flattest classes
by +0.001, which is the controlled confirmation.

Within that, two distinct failure modes were separated by the ablations:

- **`living-thing`** is a small compact object (mean connected component ~157 px). It
  responds to anything that adds effective resolution — region losses, U-Net skip
  connections, stride 8 — gaining +0.018 to +0.046 depending on the intervention.
- **`non-drivable`** is a thin strip abutting `drivable`, which is 32% of all pixels, and
  ~17-20% of it is absorbed into that neighbour. **Six independent interventions improved
  `living-thing` and made `non-drivable` worse**; only explicit class re-weighting helped
  it. Sharper, more confident models are *more* willing to call an ambiguous road edge
  "road". This is competitive absorption by a dominant adjacent class, not a resolution
  limit, and it is the failure mode closest to mattering for a drivable-area system.

**The confidence score degrades under distribution shift, and unevenly.** Temperature is
fitted on clean validation data, which is all a deployed model can do. Carrying that
temperature onto corrupted data, expected calibration error rises from 0.0026 clean to
0.0646 under severe motion blur — **25x** — while accuracy falls only to 0.82. The model
becomes wrong while staying confident, which is the failure a confidence score exists to
prevent. Test-time BatchNorm adaptation is not a general fix: it improves calibration under
motion blur (0.0646 to 0.0176) and *worsens* it under rain (0.0263 to 0.0432) while
improving rain accuracy. Accuracy recovery and calibration recovery are separate outcomes.

## Ethical considerations

Road-scene models trained on one geography routinely fail on others, and failures on the
`living-thing` class are the ones with human consequences. This model's weakest class is
exactly that one. It is published as a methods demonstration, and the out-of-scope
restriction above is a real constraint, not boilerplate.
