# Model Card — RoadSense 7-class road scene segmentation

## Model details

- **Task:** semantic segmentation, 7 classes, single label per pixel.
- **Classes:** `drivable`, `non-drivable`, `living-thing`, `vehicles`,
  `barrier-structures`, `construction-vegetation`, `sky` — IDD's own level-1 hierarchy,
  unmodified.
- **Architecture:** DeepLabV3+ decoder over a ResNet-34 encoder
  (`segmentation-models-pytorch`), ImageNet-initialised encoder.
- **Input:** RGB, resized to 320×224, ImageNet normalisation.
- **Output:** per-pixel logits; confidence is the temperature-scaled softmax maximum.
- **Serving artefact:** ONNX (opset 18), optionally INT8-quantized, run through ONNX
  Runtime on CPU.
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

## Training data

**India Driving Dataset — IDD Lite (`idd20k_lite`)**, from IIIT Hyderabad
(https://idd.insaan.iiit.ac.in/). 1607 labelled frames across 370 drive sequences at
320×227; 1403 train / 204 validation under IDD's own split.

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
- **Calibration** is fitted on one half of the validation pixels and reported on the other.
- **Robustness** uses test-time corruptions that never appear in training augmentation.
- **Reproducibility:** each run logs its config, seed and git SHA to `results/runs.csv`.

## Limitations

- Small dataset and low resolution; absolute mIoU is not comparable to full IDD 20K or
  Cityscapes results, and no such comparison is claimed.
- Corruptions are synthetic. The fog model is depth-independent because IDD Lite ships no
  depth map, so it under-represents real fog, which attenuates with distance.
- The default ablation runs one seed per cell for compute reasons; multi-seed spread is
  reported only where it was actually run.
- The architecture comparison varies the decoder within one training recipe, so it says
  nothing about architectures needing different recipes.
- Ground truth is treated as correct. No annotation-noise audit was performed.

## Ethical considerations

Road-scene models trained on one geography routinely fail on others, and failures on the
`living-thing` class are the ones with human consequences. This model's weakest class is
exactly that one. It is published as a methods demonstration, and the out-of-scope
restriction above is a real constraint, not boilerplate.
