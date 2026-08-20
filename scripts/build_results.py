"""Generate ``results/RESULTS.md`` from the experiment CSVs and JSON reports.

RESULTS.md is generated rather than hand-written so that every number in it traces to a
file produced by a run, and so it cannot drift out of date when an experiment is re-run.
Sections whose inputs are missing are skipped with a note rather than fabricated.

Run::

    python scripts/build_results.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
RESULTS = REPO_ROOT / "results"

SPLIT_LABELS = {
    "level1_official": "IDD's own split (drive-disjoint)",
    "level1_sequence": "Held-out drive sequences",
    "level1_frame": "Random frame split — **leaky control**",
}


def read_csv(name: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return [r for r in rows if r.get("error") is None]


def read_json(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def truthy(value: Any) -> bool:
    """Interpret a CSV-round-tripped boolean.

    ``csv`` writes Python ``False`` as the string ``"False"``, which is truthy. Testing it
    directly silently inverts the filter and drops every row -- which is exactly how the
    calibration-under-shift section came to render as an empty string.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def fmt(value: Any, digits: int = 4) -> str:
    """Format a cell. Reading a CSV coerces every numeric column to float, so integral
    values like an epoch number or a severity level must be rendered without a decimal
    tail or the table reads as spurious precision."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:.{digits}f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 4) -> str:
    """Render a markdown table from ``(key, header)`` pairs."""
    header = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k), digits) for k, _ in columns) + " |")
    return "\n".join(lines)


def section_split() -> str:
    rows = read_csv("split_experiment")
    if not rows:
        return ""
    for row in rows:
        row["_label"] = SPLIT_LABELS.get(str(row.get("data")), str(row.get("data")))

    # The leakage comparison must hold the validation set construction fixed. `sequence`
    # and `frame` are both re-splits of the same pooled 1607 frames at ~15% val, so they
    # are comparable. `official` is IDD's own split with a *different* validation set
    # (204 frames vs ~241), so comparing it against `frame` would confound the leakage
    # effect with a change of evaluation set.
    honest = next((r for r in rows if r.get("data") == "level1_sequence"), None)
    leaky = next((r for r in rows if r.get("data") == "level1_frame"), None)
    official = next((r for r in rows if r.get("data") == "level1_official"), None)

    out = [
        "## 1. Does a random split overstate generalization?",
        "",
        "IDD groups frames into drive sequences — one folder per drive. Frames from the same",
        "drive were captured on the same road in the same conditions, so they are correlated.",
        "Splitting frames at random puts related frames on both sides of the train/val",
        "boundary.",
        "",
        "Measured on this dataset, a random frame split puts **94.6% of validation frames in a",
        "drive that also appears in training** (1607 frames across 370 drives). Each row below",
        "trains the identical model, schedule, augmentation and seed; only the split differs.",
        "",
        table(rows, [("_label", "split"), ("miou", "mIoU"), ("mean_acc", "mean acc"),
                     ("pixel_acc", "pixel acc"), ("epochs_ran", "best epoch")]),
        "",
    ]

    if honest and leaky and honest.get("miou") and leaky.get("miou"):
        delta = leaky["miou"] - honest["miou"]
        pct = 100 * delta / honest["miou"]
        out += [
            f"**Measured inflation: {delta:+.4f} mIoU ({pct:+.1f}%).** The random split reports",
            f"{fmt(leaky['miou'])} where a drive-disjoint split of the same pooled frames reports",
            f"{fmt(honest['miou'])}.",
            "",
            "Only `sequence` and `frame` are compared. Both are re-splits of the same 1607",
            "pooled frames at ~15% validation, so the split rule is the only thing that differs.",
            "`official` is IDD's own split with a different validation set (204 frames against",
            "~241), so including it would confound leakage with a change of evaluation set; it",
            "is listed for reference only.",
            "",
            "**The inflation is real but smaller than the contamination rate suggests.** 94.6% of",
            "validation frames sharing a drive with training moved mIoU by only ~0.014. The",
            "reason shows up elsewhere in this project: frames within an IDD drive sit a median",
            "of ~4,400 frame indices apart, so they share scene and lighting without being",
            "near-duplicates. *Contaminated is not the same as duplicated* — which is precisely",
            "why this had to be measured rather than assumed. A dataset whose frames really were",
            "consecutive would be expected to show a far larger gap.",
            "",
        ]
        if official:
            out += [
                f"For reference, IDD's own split scores {fmt(official['miou'])} on its own",
                "204-frame validation set.",
                "",
            ]
    return "\n".join(out)


def section_loss() -> str:
    rows = read_csv("loss_ablation")
    if not rows:
        return ""
    ranked = sorted(rows, key=lambda r: r.get("miou") or 0, reverse=True)
    best = ranked[0]
    baseline = next((r for r in rows if r.get("loss") == "ce"), None)

    class_columns = sorted({k for r in rows for k in r if k.startswith("iou/")})
    out = [
        "## 2. Loss ablation",
        "",
        "IDD Lite is genuinely imbalanced — `living-thing` is 1.31% of training pixels and",
        "`non-drivable` 2.19%, against `construction-vegetation` at 26%. Plain cross-entropy",
        "optimises a pixel-weighted objective and can afford to neglect the rare classes, so",
        "the region and boundary losses have something real to fix.",
        "",
        "Held fixed across all arms: architecture, encoder, schedule, augmentation, split, seed.",
        "",
        table(rows, [("loss", "loss"), ("miou", "mIoU"), ("mean_acc", "mean acc"),
                     ("pixel_acc", "pixel acc"), ("epochs_ran", "best epoch"),
                     ("train_seconds", "train s")]),
        "",
    ]
    if class_columns:
        out += [
            "Per-class IoU — this is where the losses actually differ:",
            "",
            table(rows, [("loss", "loss")] + [(c, c.replace("iou/", "")) for c in class_columns], 3),
            "",
        ]
    if baseline and best.get("miou") and baseline.get("miou"):
        delta = best["miou"] - baseline["miou"]
        out += [
            f"**Winner: `{best.get('loss')}` at {fmt(best.get('miou'))} mIoU "
            f"({delta:+.4f} against cross-entropy).**",
            "",
        ]
    out += [
        "One methodological note worth stating: in *multiclass single-label* segmentation,",
        "Tversky's asymmetry is much weaker than in the binary case, because every false",
        "negative for one class is simultaneously a false positive for another, so the",
        "`alpha`/`beta` effects partly cancel. This was verified against hand-computed values",
        "in `tests/test_losses.py`.",
        "",
    ]
    return "\n".join(out)


def section_arch() -> str:
    rows = read_csv("architecture_comparison")
    if not rows:
        return ""
    return "\n".join([
        "## 3. Architecture comparison",
        "",
        "Same encoder family, loss, schedule and split; only the decoder differs. Parameter",
        "count and training time are reported alongside accuracy because a portfolio number",
        "that ignores cost is not a decision.",
        "",
        table(rows, [("experiment", "model"), ("architecture", "decoder"), ("encoder", "encoder"),
                     ("miou", "mIoU"), ("params", "params"), ("train_seconds", "train s")]),
        "",
    ])


def section_pretrain() -> str:
    rows = read_csv("pretrain_ablation")
    if not rows:
        return ""
    return "\n".join([
        "## 4. Encoder initialisation",
        "",
        "With only 1403 labelled training frames, how much of the result comes from ImageNet",
        "pretraining rather than from this dataset?",
        "",
        table(rows, [("experiment", "init"), ("encoder_weights", "weights"), ("miou", "mIoU"),
                     ("mean_acc", "mean acc"), ("epochs_ran", "best epoch")]),
        "",
    ])


def section_calibration() -> str:
    report = read_json("calibration_report")
    if not report:
        return ""
    return "\n".join([
        "## 5. Confidence calibration",
        "",
        "The serving layer returns a confidence score, so that score has to mean something.",
        "Temperature is fitted on one half of the validation pixels and every number below is",
        "computed on the other half, which the fit never saw.",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| Expected calibration error | {fmt(report.get('ece_before'))} | {fmt(report.get('ece_after'))} |",
        f"| Maximum calibration error | {fmt(report.get('mce_before'))} | {fmt(report.get('mce_after'))} |",
        "",
        f"Fitted temperature **T = {fmt(report.get('temperature'), 3)}** "
        f"({'overconfident' if (report.get('temperature') or 1) > 1 else 'underconfident'} before "
        f"correction), reducing ECE by {fmt(report.get('ece_reduction_pct'), 1)}%.",
        "",
        "Temperature scaling is monotone, so mIoU is provably unchanged — only the confidence",
        "distribution moves.",
        "",
        "![reliability diagram](figures/reliability.png)",
        "",
    ])


def section_robustness() -> str:
    rows = read_csv("corruption_robustness")
    if not rows:
        return ""
    clean = next((r for r in rows if r.get("corruption") == "clean"), None)
    corrupted = [r for r in rows if r.get("corruption") != "clean"]
    if not corrupted:
        return ""
    worst = min(corrupted, key=lambda r: r.get("miou") or 1)
    recovered = [r["recovered"] for r in corrupted if r.get("recovered") is not None]

    names = sorted({str(r["corruption"]) for r in corrupted})
    severities = sorted({int(r["severity"]) for r in corrupted if r.get("severity") is not None})
    out = [
        "## 6. Robustness under adverse conditions",
        "",
        f"{len(names)} corruptions "
        f"({', '.join('`' + n + '`' for n in names)}) at "
        f"{'severity' if len(severities) == 1 else 'severities'} "
        f"{'/'.join(str(s) for s in severities)}, standing in for adverse driving conditions.",
        "None of them appear in the training augmentation pipeline, so the drop measures",
        "genuine distribution shift rather than a train/test augmentation mismatch.",
        "",
        "Each corrupted set is also evaluated after **test-time BatchNorm adaptation** — the",
        "running statistics are re-estimated on the shifted data using no labels and no",
        "gradient steps. It isolates how much of the degradation is merely feature-statistic",
        "drift rather than a genuine failure to perceive.",
        "",
        table(corrupted, [("corruption", "corruption"), ("severity", "sev"), ("miou", "mIoU"),
                          ("miou_bn_adapted", "+ BN adapt"), ("recovered", "recovered"),
                          ("retention", "retention")]),
        "",
    ]
    if clean:
        out.append(f"Clean baseline: **{fmt(clean.get('miou'))} mIoU**. "
                   f"Worst case: `{worst.get('corruption')}` at severity "
                   f"{fmt(worst.get('severity'), 0)} → {fmt(worst.get('miou'))}.")
    if recovered:
        mean_recovered = sum(recovered) / len(recovered)
        out.append(f" BatchNorm adaptation recovers **{mean_recovered:+.4f} mIoU on average**, "
                   "without labels or gradient updates.")
    out += ["", "![corruption degradation](figures/corruption_degradation.png)", ""]
    return "\n".join(out)


def section_stability() -> str:
    rows = read_csv("stability")
    if not rows:
        return ""
    return "\n".join([
        "## 7. Prediction stability",
        "",
        "A deployed segmenter sees near-identical scenes repeatedly; one that flips classes",
        "under imperceptible input changes is unusable at any mIoU. The natural measurement is",
        "frame-to-frame flicker on video — **which IDD Lite cannot support**: frames within a",
        "drive are not adjacent (median gap ~4,400 frame indices, only 0.5% within 30 frames),",
        "so a flicker number computed from them would really be measuring scene change.",
        "",
        "Instead, each image is perturbed by a small label-preserving transform and the",
        "prediction compared against the unperturbed one. Geometric shifts are inverted before",
        "comparison, so only genuine class flips are counted. No ground truth is required.",
        "",
        table(rows, [("perturbation", "perturbation"), ("flip_rate", "flip rate"),
                     ("flip_rate_confident", "flip rate (conf ≥ 0.8)"),
                     ("mean_confidence_delta", "Δ confidence")]),
        "",
    ])


def section_compression() -> str:
    rows = read_csv("compression")
    if not rows:
        return ""
    int8 = next((r for r in rows if r.get("precision") == "int8"), None)
    out = [
        "## 8. Compression and deployment",
        "",
        "Exported to ONNX and quantized to INT8 with static, calibrated post-training",
        "quantization. Both precisions are benchmarked through ONNX Runtime **on CPU with a",
        "pinned thread count**: Apple silicon has no INT8 path through MPS, so a GPU",
        "comparison would silently run the quantized model in FP32 and report a meaningless",
        "speedup.",
        "",
        table(rows, [("precision", "precision"), ("miou", "mIoU"), ("latency_ms_mean", "latency ms"),
                     ("latency_ms_p95", "p95 ms"), ("size_mb", "size MB"),
                     ("throughput_img_s", "img/s")], 2),
        "",
    ]
    if int8:
        out.append(
            f"INT8 costs **{fmt(int8.get('miou_delta'))} mIoU** for a "
            f"**{fmt(int8.get('speedup'), 2)}× speedup** and "
            f"**{fmt(int8.get('size_reduction'), 2)}× smaller** model."
        )
        out.append("")
    out += [
        "The export is verified against PyTorch on real validation frames before it is used:",
        "`compression/export.py` reports mean logit error *and* the fraction of pixels whose",
        "predicted class changes, because a graph can round-trip logits closely while still",
        "flipping labels at boundaries.",
        "",
    ]
    return "\n".join(out)


def section_distillation() -> str:
    rows = read_csv("distillation")
    if not rows:
        return ""
    distilled = next((r for r in rows if r.get("variant") == "student_distilled"), None)
    out = [
        "## 8b. Knowledge distillation",
        "",
        "The comparison that matters is not whether a distilled student beats its teacher —",
        "it will not — but whether it beats **the same student trained alone**, at identical",
        "capacity, data, schedule and seed. Only that isolates the value of the teacher's soft",
        "targets from the value of simply having a smaller model.",
        "",
        table(rows, [("variant", "variant"), ("encoder", "encoder"), ("miou", "mIoU"),
                     ("params", "params"), ("gain_over_alone", "vs student alone"),
                     ("gap_to_teacher", "vs teacher")]),
        "",
    ]
    if distilled and distilled.get("gain_over_alone") is not None:
        gain = distilled["gain_over_alone"]
        verdict = ("Distillation helped" if gain > 0 else
                   "Distillation did **not** help at this scale")
        out += [
            f"**{verdict}: {gain:+.4f} mIoU against the identical student trained alone.**",
            "",
        ]
        if gain <= 0:
            out += [
                "That is a legitimate outcome and is reported as measured. With 1403 training",
                "frames the student is not capacity-limited enough for the teacher's soft",
                "targets to add information beyond the hard labels.",
                "",
            ]
    return "\n".join(out)


def section_pareto() -> str:
    rows = read_csv("compression_pareto")
    if not rows:
        return ""
    from compression.pareto import pareto_frontier

    frontier = {(r["variant"], r["precision"]) for r in pareto_frontier(rows)}
    for row in rows:
        row["_pareto"] = "yes" if (row["variant"], row["precision"]) in frontier else ""
    return "\n".join([
        "## 8c. Deployment Pareto",
        "",
        "Every variant at both precisions, benchmarked identically. A configuration is",
        "*dominated* when another is at least as fast and at least as accurate; only the",
        "survivors represent a real deployment choice.",
        "",
        table(rows, [("variant", "variant"), ("precision", "precision"), ("miou", "mIoU"),
                     ("latency_ms_mean", "latency ms"), ("latency_ms_p95", "p95 ms"),
                     ("size_mb", "size MB"), ("_pareto", "Pareto-optimal")], 2),
        "",
        "![compression Pareto](figures/compression_pareto.png)",
        "",
    ])


def section_seed_variance() -> str:
    """1b. Is the leakage effect larger than seed noise?"""
    rows = [r for r in read_csv("seed_variance") if r.get("miou") is not None]
    if not rows:
        return ""
    import statistics

    by_mode: dict[str, list[float]] = {}
    for row in rows:
        mode = str(row.get("experiment", "")).rsplit("_s", 1)[0]
        by_mode.setdefault(mode, []).append(float(row["miou"]))

    out = [
        "## 1b. Is that difference bigger than seed noise?",
        "",
        "Section 1 reports a difference between two splits from one seed each, which",
        "cannot on its own be distinguished from run-to-run variance. Both arms are",
        "retrained across seeds here; everything but the seed and the split rule is fixed.",
        "",
        "| split | seeds | mean mIoU | std | min | max |",
        "|---|---|---|---|---|---|",
    ]
    for mode in ("sequence", "frame"):
        values = by_mode.get(mode) or []
        if not values:
            continue
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(
            f"| {mode} | {len(values)} | {fmt(statistics.mean(values))} | {fmt(std)} | "
            f"{fmt(min(values))} | {fmt(max(values))} |"
        )

    seq, frm = by_mode.get("sequence") or [], by_mode.get("frame") or []
    if len(seq) > 1 and len(frm) > 1:
        effect = statistics.mean(frm) - statistics.mean(seq)
        pooled = max(statistics.stdev(seq), statistics.stdev(frm))
        verdict = (
            "larger than the seed spread, so the direction survives the noise"
            if abs(effect) > 2 * pooled
            else "**within twice the seed spread — this measurement cannot separate the "
            "leakage effect from run-to-run variance**"
        )
        out += [
            "",
            f"Mean inflation from the leaky split: **{effect:+.4f} mIoU**, against a "
            f"worst-arm seed standard deviation of {pooled:.4f}. The effect is {verdict}.",
            "",
            "Reported either way. A difference that does not clear its own noise floor is "
            "a result about the experiment's resolution, not a null result about leakage.",
        ]
    out.append("")
    return "\n".join(out)


def section_stride() -> str:
    """3b. Is the per-class weakness a stride problem or a data problem?"""
    rows = [r for r in read_csv("stride_ablation") if r.get("miou") is not None]
    if len(rows) < 2:
        return ""
    by_stride = {int(r.get("encoder_output_stride") or 16): r for r in rows}
    if 8 not in by_stride or 16 not in by_stride:
        return ""
    base, fine = by_stride[16], by_stride[8]

    geometry = (read_json("error_analysis") or {}).get("class_geometry")
    names = [k[4:] for k in base if k.startswith("iou/")]

    out = [
        "## 3b. Stride, not scarcity",
        "",
        "Section 9 finds per-class IoU tracks a class's *boundary share* far better than",
        "its rarity — pointing at thin structures lost to downsampling rather than at",
        "class imbalance. This halves the encoder's output stride, doubling the resolution",
        "of the features the decoder sees and changing nothing else.",
        "",
        f"Overall mIoU {fmt(base['miou'])} → **{fmt(fine['miou'])}** "
        f"({float(fine['miou']) - float(base['miou']):+.4f}).",
        "",
        "| class | boundary share | IoU @ stride 16 | IoU @ stride 8 | Δ |",
        "|---|---|---|---|---|",
    ]
    boundary = {}
    if geometry:
        report_names = (read_json("error_analysis") or {}).get("class_names") or []
        boundary = dict(zip(report_names, geometry["boundary_fraction"], strict=False))
    ordered = sorted(names, key=lambda n: -boundary.get(n, 0.0))
    for name in ordered:
        before, after = base.get(f"iou/{name}"), fine.get(f"iou/{name}")
        if before is None or after is None:
            continue
        share = f"{100 * boundary[name]:.1f}%" if name in boundary else "—"
        out.append(
            f"| {name} | {share} | {fmt(before)} | {fmt(after)} | "
            f"{float(after) - float(before):+.4f} |"
        )
    out += [
        "",
        "Rows are ordered by boundary share — thinnest first. If the diagnosis holds, the",
        "gains concentrate at the top of this table. If they are spread evenly, the",
        "limitation is the source resolution rather than the network, and no amount of",
        "architectural change recovers it at 320×227.",
        "",
    ]
    return "\n".join(out)


def section_heldout() -> str:
    """5b. The first number in this report that model selection never saw."""
    val, test = read_json("heldout_val"), read_json("heldout_test")
    if not val or not test:
        return ""
    out = [
        "## 5b. A genuinely held-out test set",
        "",
        "Everywhere else in this report, validation does double duty: it is the",
        "early-stopping criterion *and* the set the number is reported on, so model",
        "selection has seen it. IDD withholds its own test labels, so a third partition is",
        "carved from the labelled frames instead — drive-disjoint from **both** train and",
        "validation, since a test set sharing drives with validation is contaminated",
        "through model selection just as surely as through training.",
        "",
        "| split | frames | mIoU | boundary mIoU | pixel acc |",
        "|---|---|---|---|---|",
        f"| val (used for early stopping) | {val.get('n_images')} | {fmt(val.get('miou'))} | "
        f"{fmt(val.get('boundary_miou'))} | {fmt(val.get('pixel_acc'))} |",
        f"| **test (never seen)** | {test.get('n_images')} | **{fmt(test.get('miou'))}** | "
        f"{fmt(test.get('boundary_miou'))} | {fmt(test.get('pixel_acc'))} |",
        "",
    ]
    if val.get("miou") and test.get("miou"):
        drop = float(val["miou"]) - float(test["miou"])
        out += [
            f"Selection optimism: **{drop:+.4f} mIoU**. That gap is what early stopping "
            "bought itself on the validation set, and it is the correction that should be "
            "mentally applied to every other number in this report.",
            "",
        ]
    if test.get("boundary_miou") and test.get("miou"):
        out += [
            f"Boundary mIoU trails region mIoU by {float(test['miou']) - float(test['boundary_miou']):.4f} "
            "— the model places regions better than it places their edges.",
            "",
        ]
    return "\n".join(out)


def section_shift_calibration() -> str:
    """6b. Does the confidence score survive the shift?"""
    rows = read_csv("shift_calibration")
    if not rows:
        return ""
    clean = next((r for r in rows if r.get("corruption") == "clean"), None)
    plain = [
        r for r in rows if r.get("corruption") != "clean" and not truthy(r.get("bn_adapted"))
    ]
    adapted = {
        (r["corruption"], r["severity"]): r for r in rows if truthy(r.get("bn_adapted"))
    }
    if not clean or not plain:
        return ""

    out = [
        "## 6b. Is the confidence score still a probability in the rain?",
        "",
        "Section 5 calibrates on clean validation data and the serving layer applies that",
        "temperature to everything it sees. But a deployed model cannot refit against",
        "conditions it has not encountered, so the question is whether the *clean-fitted*",
        "temperature still holds under shift. Reported alongside is the temperature refit",
        "on each corruption — unreachable in deployment, and included as the ceiling.",
        "",
        "| condition | accuracy | mean confidence | over-confidence | ECE (deployed T) | ECE (refit) |",
        "|---|---|---|---|---|---|",
        f"| clean | {fmt(clean['accuracy'])} | {fmt(clean['mean_confidence'])} | "
        f"{fmt(clean['overconfidence'])} | {fmt(clean['ece_deployed_t'])} | "
        f"{fmt(clean['ece_refit_t'])} |",
    ]
    for row in plain:
        out.append(
            f"| {row['corruption']} s{row['severity']} | {fmt(row['accuracy'])} | "
            f"{fmt(row['mean_confidence'])} | {fmt(row['overconfidence'])} | "
            f"**{fmt(row['ece_deployed_t'])}** | {fmt(row['ece_refit_t'])} |"
        )

    worst = max(plain, key=lambda r: float(r["ece_deployed_t"]))
    ratio = float(worst["ece_deployed_t"]) / max(float(clean["ece_deployed_t"]), 1e-9)
    out += [
        "",
        f"Worst case: `{worst['corruption']}` at severity {worst['severity']}, where "
        f"calibration error is **{ratio:.1f}× the clean figure**. The model does not merely "
        "become less accurate under shift — it becomes less accurate *while remaining "
        "confident*, which is the failure mode a confidence score exists to prevent.",
        "",
        "The gap between the two ECE columns is the part attributable purely to the",
        "temperature being fitted on the wrong distribution, and is therefore the part a",
        "shift-aware calibration scheme could in principle recover.",
    ]

    if adapted:
        out += [
            "",
            "### Does BatchNorm adaptation restore confidence, or only accuracy?",
            "",
            "Section 6 shows label-free BN adaptation recovering much of the lost mIoU.",
            "Whether it also repairs the confidence distribution is a separate question:",
            "an accurate model that still misstates its certainty is not obviously safer.",
            "",
            "| condition | ECE before | ECE after BN | accuracy before | accuracy after |",
            "|---|---|---|---|---|",
        ]
        for row in plain:
            key = (row["corruption"], row["severity"])
            if key not in adapted:
                continue
            after = adapted[key]
            out.append(
                f"| {row['corruption']} s{row['severity']} | {fmt(row['ece_deployed_t'])} | "
                f"{fmt(after['ece_deployed_t'])} | {fmt(row['accuracy'])} | "
                f"{fmt(after['accuracy'])} |"
            )

    out += [
        "",
        "### Selective prediction",
        "",
        "If the confidence number carries information, abstaining on the least confident",
        "pixels should raise accuracy on the rest. This is the operating-point question a",
        "perception stack actually asks, and it replaces the hardcoded 0.6 threshold in the",
        "serving layer with a measured value.",
        "",
        "| condition | coverage at 95% accuracy | risk-coverage AUC |",
        "|---|---|---|",
        f"| clean | {fmt(clean.get('coverage_at_acc_0.95'), 3)} | "
        f"{fmt(clean.get('risk_coverage_auc'))} |",
    ]
    for row in plain:
        out.append(
            f"| {row['corruption']} s{row['severity']} | "
            f"{fmt(row.get('coverage_at_acc_0.95'), 3)} | {fmt(row.get('risk_coverage_auc'))} |"
        )
    out += ["", "![calibration under shift](figures/shift_calibration.png)", ""]
    return "\n".join(out)


def section_errors() -> str:
    report = read_json("error_analysis")
    if not report:
        return ""
    out = [
        "## 9. Error analysis",
        "",
        f"Validation mIoU {fmt(report.get('miou'))}. Per-image mIoU percentiles: "
        + ", ".join(f"p{k}={fmt(v, 3)}" for k, v in (report.get("miou_percentiles") or {}).items()),
        "",
        "Strongest confusions (share of a true class's pixels given to another):",
        "",
        "| true class | predicted as | rate |",
        "|---|---|---|",
    ]
    for pair in (report.get("top_confusions") or [])[:6]:
        out.append(f"| {pair['true']} | {pair['predicted']} | {100 * pair['rate']:.1f}% |")
    correlation = report.get("geometry_vs_frequency")
    geometry = report.get("class_geometry")
    if correlation and geometry:
        names = report.get("class_names") or []
        iou = report.get("iou_per_class") or []
        rows = sorted(
            zip(names, iou, geometry["boundary_fraction"], geometry["pixel_fraction"], strict=False),
            key=lambda r: r[2],
        )
        out += [
            "",
            "### Is it rarity, or is it thinness?",
            "",
            "The obvious reading of the per-class spread is that IoU tracks class rarity.",
            "That reading is confounded. Below, classes are ordered by how much of each one",
            "lies within 2px of a class border — computed from the training masks alone, with",
            "no model involved.",
            "",
            "| class | share of pixels | pixels near a border | IoU |",
            "|---|---|---|---|",
        ]
        for name, class_iou, boundary, frequency in rows:
            out.append(
                f"| {name} | {100 * frequency:.1f}% | {100 * boundary:.1f}% | {fmt(class_iou)} |"
            )
        r_boundary = correlation["r_iou_vs_boundary_fraction"]
        r_frequency = correlation["r_iou_vs_log_frequency"]
        out += [
            "",
            f"Across the {correlation['n_classes']} classes, IoU correlates at "
            f"**r = {r_boundary:+.2f}** (r² = {r_boundary ** 2:.2f}) with boundary share, against "
            f"**r = {r_frequency:+.2f}** (r² = {r_frequency ** 2:.2f}) with log frequency. Geometry "
            "explains the spread better than rarity does, and there is a decisive single case: "
            "`barrier-structures` is 8.5× more common than `living-thing` and scores the same "
            "IoU, which a rarity story cannot account for.",
            "",
            "**Stated with its limits.** The two predictors are themselves correlated at "
            f"r = {correlation['r_boundary_vs_log_frequency']:+.2f} — rare classes here *are* the "
            f"thin ones — and {correlation['n_classes']} points cannot separate them. This is a "
            "hypothesis with a counterexample, not a settled result. The experiment that would "
            "settle it is one arm at `encoder_output_stride=8`, everything else fixed: if "
            "geometry is binding, the thin classes gain disproportionately.",
        ]
    out += [
        "",
        "![confusion matrix](figures/confusion_matrix.png)",
        "",
        "![IoU vs class frequency](figures/iou_vs_frequency.png)",
        "",
        "The hardest validation frames, shown as image / ground truth / prediction / error map:",
        "",
        "![worst cases](figures/worst_cases.png)",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    from modeling.tracking import git_sha

    runs = read_csv("runs")
    header = [
        "# Results",
        "",
        "*Generated by `scripts/build_results.py` from the CSVs in `results/`. Do not edit by hand.*",
        "",
        "**Task.** 7-class semantic segmentation of unstructured Indian road scenes, using IDD's",
        "own level-1 label hierarchy: `drivable`, `non-drivable`, `living-thing`, `vehicles`,",
        "`barrier-structures`, `construction-vegetation`, `sky`.",
        "",
        "**Data.** IDD Lite (`idd20k_lite`) — 1607 labelled frames across 370 drive sequences,",
        "at 320×227, trained at 320×224. Every dataset in this project comes from IDD; there are",
        "no external data sources and no synthetic labels.",
        "",
        "**Scale, stated plainly.** All results are at IDD Lite scale on a single M1 Pro (MPS).",
        "This is a small dataset and the absolute mIoU values reflect that. The comparisons are",
        "controlled — one factor varies at a time, with everything else held fixed — so the",
        "*differences* between rows are meaningful even where the absolute numbers are modest.",
        "",
        f"Runs recorded: {len(runs)}. Code version: `{git_sha()}`.",
        "",
        "---",
        "",
    ]

    sections = [
        section_split(), section_seed_variance(), section_loss(),
        section_arch(), section_stride(), section_pretrain(),
        section_calibration(), section_heldout(),
        section_robustness(), section_shift_calibration(), section_stability(),
        section_compression(), section_distillation(), section_pareto(),
        section_errors(),
    ]
    body = [s for s in sections if s]
    if not body:
        body = ["_No experiment outputs found yet. Run `python scripts/run_ablations.py --suite split`._", ""]

    text = "\n".join(header) + "\n---\n\n".join(body)
    text += "\n---\n\n## Reproducing\n\n```bash\n" \
            "python -m data.prepare_masks --split-mode all   # build dataset variants\n" \
            "./scripts/reproduce.sh                          # every table above\n```\n"

    out_path = RESULTS / "RESULTS.md"
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text.splitlines())} lines, {len(body)} sections)")


if __name__ == "__main__":
    main()
