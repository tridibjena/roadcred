"""Generate ``results/RESULTS.md`` from the experiment CSVs and JSON reports.

RESULTS.md is generated rather than hand-written so that every number in it traces to a
file produced by a run, and so it cannot drift out of date when an experiment is re-run.
That includes the dataset facts quoted in prose -- frame counts, the leakage rate, the
gap between frames of a drive -- which come from ``results/dataset_stats.json``, written
by ``python -m data.prepare_masks``. Sections whose inputs are missing are skipped rather
than fabricated.

Run::

    python scripts/build_results.py
"""

from __future__ import annotations

import csv
import json
import statistics
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

#: Every logged setting that defines a training run. Rows of ``runs.csv`` agreeing on all
#: of them -- seed included -- are repeats of one experiment, so their spread is pure
#: run-to-run noise.
IDENTITY_KEYS = (
    "data", "architecture", "encoder", "encoder_weights", "encoder_output_stride", "loss",
    "seed", "batch_size", "epochs_max", "lr", "weight_decay", "imgsz", "teacher",
)


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


def whole_seconds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Training time is wall-clock; a fractional second is noise, not information."""
    return [
        {**r, "train_seconds": float(round(r["train_seconds"]))}
        if r.get("train_seconds") is not None else r
        for r in rows
    ]


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


# --- shared measurements ----------------------------------------------------------------


def dataset_stats() -> dict[str, Any]:
    return read_json("dataset_stats") or {}


def class_shares() -> dict[str, float]:
    """Training-pixel share per class, from the error analysis' geometry pass."""
    report = read_json("error_analysis") or {}
    geometry = report.get("class_geometry") or {}
    return dict(zip(report.get("class_names") or [], geometry.get("pixel_fraction") or [],
                    strict=False))


def _identity(row: dict[str, Any]) -> tuple:
    def norm(key: str) -> Any:
        value = row.get(key)
        if key == "encoder_output_stride":
            return float(value) if value is not None else 16.0  # unset = smp's default
        return value

    return tuple(norm(k) for k in IDENTITY_KEYS)


def identical_run_groups(runs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Groups of runs that share every logged setting, seed included."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in runs:
        if row.get("miou") is not None:
            groups.setdefault(_identity(row), []).append(row)
    return sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)


def baseline_group(runs: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """The repeated DeepLabV3+/ResNet-34 cross-entropy run on IDD's own split -- the
    baseline every single-seed intervention in this report is compared against."""
    for group in identical_run_groups(runs):
        row = group[0]
        if (row.get("data") == "level1_official" and row.get("architecture") == "deeplabv3plus"
                and row.get("encoder") == "resnet34" and row.get("loss") == "ce"
                and _identity(row)[IDENTITY_KEYS.index("encoder_output_stride")] == 16.0):
            return group
    return None


def band(group: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    values = [float(r[key]) for r in group if r.get(key) is not None]
    return (min(values), max(values)) if len(values) > 1 else None


def mean_of(group: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(r[key]) for r in group if r.get(key) is not None)


# --- sections ---------------------------------------------------------------------------


def section_split() -> str:
    rows = read_csv("split_experiment")
    if not rows:
        return ""
    stats = dataset_stats()
    splits = stats.get("splits") or {}
    leak = (splits.get("frame") or {}).get("contaminated_fraction")
    gaps = stats.get("frame_gaps") or {}
    for row in rows:
        row["_label"] = SPLIT_LABELS.get(str(row.get("data")), str(row.get("data")))

    # The leakage comparison must hold the validation set construction fixed. `sequence`
    # and `frame` are both re-splits of the same pooled frames at the same validation
    # fraction, so they are comparable. `official` is IDD's own split with a *different*
    # validation set, so comparing it against `frame` would confound the leakage effect
    # with a change of evaluation set.
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
    ]
    if leak is not None:
        out += [
            f"Measured on this dataset, a random frame split puts **{pct(leak)} of validation "
            "frames in a drive that also appears in training** "
            f"({stats.get('n_frames')} frames across {stats.get('n_drives')} drives). Each row "
            "below trains the identical model, schedule, augmentation and seed; only the split "
            "differs.",
            "",
        ]
    out += [
        table(rows, [("_label", "split"), ("miou", "mIoU"), ("mean_acc", "mean acc"),
                     ("pixel_acc", "pixel acc"), ("epochs_ran", "best epoch")]),
        "",
    ]

    if honest and leaky and honest.get("miou") and leaky.get("miou"):
        delta = leaky["miou"] - honest["miou"]
        n_val = {mode: (splits.get(mode) or {}).get("n_val") for mode in splits}
        out += [
            f"**Measured inflation, one seed per arm: {delta:+.4f} mIoU "
            f"({100 * delta / honest['miou']:+.1f}%).** §1b repeats both arms across seeds.",
            "",
            "Only `sequence` and `frame` are compared: both re-split the same pooled frames "
            f"({n_val.get('sequence')} and {n_val.get('frame')} validation frames), so the split "
            f"rule is the only difference. `official` validates on IDD's own "
            f"{n_val.get('official')} frames, a different set, so it is listed for reference "
            "only.",
            "",
        ]
        if leak is not None and gaps.get("median_gap"):
            within = gaps.get("share_within_30")
            out += [
                f"**The inflation is real but small next to the contamination rate.** "
                f"{pct(leak)} of validation frames sharing a drive with training moved mIoU by "
                f"{delta:.3f}. Frames within an IDD drive sit a median of "
                f"{fmt(float(gaps['median_gap']), 0)} frame indices apart"
                + (f", and only {pct(within)} of consecutive gaps are within 30 frames"
                   if within is not None else "")
                + ", so they share scene and lighting without being near-duplicates. "
                "*Contaminated is not the same as duplicated* — which is why this had to be "
                "measured rather than assumed.",
                "",
            ]
        if official:
            out += [f"For reference, IDD's own split scores {fmt(official['miou'])}.", ""]
    return "\n".join(out)


def section_seed_variance() -> str:
    """1b. Is the leakage effect larger than seed noise?"""
    rows = [r for r in read_csv("seed_variance") if r.get("miou") is not None]
    if not rows:
        return ""
    by_mode: dict[str, list[float]] = {}
    for row in rows:
        mode = str(row.get("experiment", "")).rsplit("_s", 1)[0]
        by_mode.setdefault(mode, []).append(float(row["miou"]))

    out = [
        "## 1b. Is that difference bigger than seed noise?",
        "",
        "Section 1 compares one seed per split. Here both arms are retrained across seeds;",
        "everything but the seed and the split rule is fixed. All seeds train on the same",
        "split (built once, with split seed 0), so this samples training noise, not the",
        "variation from which drives land in validation.",
        "",
        "| split | seeds | mean mIoU | std | min | max |",
        "|---|---|---|---|---|---|",
    ]
    for mode in ("sequence", "frame"):
        values = by_mode.get(mode) or []
        if values:
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            out.append(
                f"| {mode} | {len(values)} | {fmt(statistics.mean(values))} | {fmt(std)} | "
                f"{fmt(min(values))} | {fmt(max(values))} |"
            )

    seq, frm = by_mode.get("sequence") or [], by_mode.get("frame") or []
    if len(seq) > 1 and len(frm) > 1:
        from scipy import stats

        effect = statistics.mean(frm) - statistics.mean(seq)
        welch = stats.ttest_ind(frm, seq, equal_var=False)
        s_seq, s_frm = statistics.stdev(seq), statistics.stdev(frm)
        ratio = (max(s_seq, s_frm) / min(s_seq, s_frm)) ** 2 if min(s_seq, s_frm) else float("inf")
        df_high = (len(seq) if s_seq >= s_frm else len(frm)) - 1
        df_low = (len(frm) if s_seq >= s_frm else len(seq)) - 1
        p_var = min(1.0, 2 * stats.f.sf(ratio, df_high, df_low))
        verdict = (
            "**the leakage effect is statistically resolved**" if welch.pvalue < 0.05
            else "**not resolved at this number of seeds**"
        )
        out += [
            "",
            f"Mean inflation from the leaky split: **{effect:+.4f} mIoU** (Welch's t = "
            f"{welch.statistic:.2f}, p = {welch.pvalue:.3f}) — {verdict}.",
            "",
            f"The two arms' spreads differ ({fmt(s_frm)} for `frame`, {fmt(s_seq)} for "
            f"`sequence`), but with {len(seq)} and {len(frm)} seeds that difference is not "
            f"distinguishable from chance (F-test on the variance ratio, p = {p_var:.2f}). It "
            "is an observation, not a finding.",
        ]
    out.append("")
    return "\n".join(out)


def section_noise_floor() -> str:
    """1c. How far apart are runs that share every setting, seed included?"""
    runs = read_csv("runs")
    groups = identical_run_groups(runs)
    if not groups:
        return ""
    out = [
        "## 1c. How far apart are identical runs?",
        "",
        "MPS offers no determinism guarantee (`cudnn.deterministic` is CUDA-only), so runs",
        "with the same configuration *and* seed still disagree. Several suites repeat a",
        "configuration exactly; within each group below, every logged setting matches.",
        "",
        "| data | model | loss | seed | runs | mIoU range |",
        "|---|---|---|---|---|---|",
    ]
    for group in groups:
        row, (low, high) = group[0], band(group, "miou")
        out.append(f"| {row['data']} | {row['architecture']}/{row['encoder']} | {row['loss']} | "
                   f"{fmt(row['seed'])} | {len(group)} | {high - low:.4f} |")

    base = baseline_group(runs)
    if base:
        classes = sorted(k for k in base[0] if k.startswith("iou/"))
        out += [
            "",
            "Per-class IoU is far noisier than mIoU. Across the "
            f"{len(base)} identical baseline runs on IDD's own split:",
            "",
            "| class | min | max | range |",
            "|---|---|---|---|",
        ]
        for key in classes:
            limits = band(base, key)
            if limits:
                out.append(f"| {key[4:]} | {limits[0]:.4f} | {limits[1]:.4f} | "
                           f"{limits[1] - limits[0]:.4f} |")
        out += [
            "",
            "A single-seed difference inside these ranges is unresolved. §3b marks which",
            "per-class changes clear them.",
        ]
    out.append("")
    return "\n".join(out)


def section_loss() -> str:
    rows = read_csv("loss_ablation")
    if not rows:
        return ""
    ranked = sorted(rows, key=lambda r: r.get("miou") or 0, reverse=True)
    best = ranked[0]
    baseline = next((r for r in rows if r.get("loss") == "ce"), None)
    shares = class_shares()

    class_columns = sorted({k for r in rows for k in r if k.startswith("iou/")})
    out = ["## 2. Loss ablation", ""]
    if {"living-thing", "non-drivable", "construction-vegetation"} <= set(shares):
        out += [
            f"IDD Lite is genuinely imbalanced — `living-thing` is {pct(shares['living-thing'], 2)} "
            f"of training pixels and `non-drivable` {pct(shares['non-drivable'], 2)}, against "
            f"`construction-vegetation` at {pct(shares['construction-vegetation'], 0)}. Plain "
            "cross-entropy optimises a pixel-weighted objective and can afford to neglect the "
            "rare classes, so the region and boundary losses have something real to fix.",
            "",
        ]
    out += [
        "Held fixed across all arms: architecture, encoder, schedule, augmentation, split, seed.",
        "",
        table(whole_seconds(rows), [("loss", "loss"), ("miou", "mIoU"), ("mean_acc", "mean acc"),
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
            f"**Highest: `{best.get('loss')}` at {fmt(best.get('miou'))} mIoU "
            f"({delta:+.4f} against cross-entropy, one seed each).**",
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
        table(whole_seconds(rows), [("experiment", "model"), ("architecture", "decoder"),
                                    ("encoder", "encoder"), ("miou", "mIoU"),
                                    ("params", "params"), ("train_seconds", "train s")]),
        "",
    ])


def _interventions() -> list[tuple[str, dict[str, Any]]]:
    """Every single-seed change made to the baseline on IDD's own split."""
    found = [(f"loss: {r['loss']}", r) for r in read_csv("loss_ablation") if r.get("loss") != "ce"]
    found += [
        (f"decoder: {r['architecture']}", r) for r in read_csv("architecture_comparison")
        if r.get("architecture") in ("unet", "fpn") and r.get("encoder") == "resnet34"
    ]
    found += [
        ("encoder stride 8", r) for r in read_csv("stride_ablation")
        if r.get("encoder_output_stride") == 8
    ]
    return found


def section_stride() -> str:
    """3b. Is the per-class weakness a stride problem or a data problem?"""
    rows = [r for r in read_csv("stride_ablation") if r.get("miou") is not None]
    if len(rows) < 2:
        return ""
    by_stride = {int(r.get("encoder_output_stride") or 16): r for r in rows}
    if 8 not in by_stride or 16 not in by_stride:
        return ""
    base, fine = by_stride[16], by_stride[8]
    noise = baseline_group(read_csv("runs")) or []

    report = read_json("error_analysis") or {}
    geometry = report.get("class_geometry")
    boundary = (
        dict(zip(report.get("class_names") or [], geometry["boundary_fraction"], strict=False))
        if geometry else {}
    )
    names = sorted((k[4:] for k in base if k.startswith("iou/")),
                   key=lambda n: -boundary.get(n, 0.0))

    out = [
        "## 3b. Stride, not scarcity",
        "",
        "Section 9 finds per-class IoU tracks a class's *boundary share* far better than its",
        "rarity — pointing at thin structures lost to downsampling rather than at class",
        "imbalance. This halves the encoder's output stride, doubling the resolution of the",
        "features the decoder sees and changing nothing else.",
        "",
        f"Overall mIoU {fmt(base['miou'])} → **{fmt(fine['miou'])}** "
        f"({float(fine['miou']) - float(base['miou']):+.4f})"
        + (f", against an identical-run mIoU range of {band(noise, 'miou')[1] - band(noise, 'miou')[0]:.4f} (§1c)."
           if band(noise, "miou") else "."),
        "",
        "| class | boundary share | IoU @ stride 16 | IoU @ stride 8 | Δ | identical-run range | outside it? |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        before, after = base.get(f"iou/{name}"), fine.get(f"iou/{name}")
        if before is None or after is None:
            continue
        limits = band(noise, f"iou/{name}")
        share = pct(boundary[name]) if name in boundary else "—"
        spread = f"{limits[1] - limits[0]:.4f}" if limits else "—"
        clears = ("yes" if not limits[0] <= float(after) <= limits[1] else "no") if limits else "—"
        out.append(f"| {name} | {share} | {fmt(before)} | {fmt(after)} | "
                   f"{float(after) - float(before):+.4f} | {spread} | {clears} |")
    out += [
        "",
        "Rows are ordered by boundary share — thinnest first. \"Outside it?\" asks whether the",
        "stride-8 IoU falls outside the range the identical baseline runs span; a change",
        "inside that range is unresolved at one seed.",
        "",
    ]

    interventions = _interventions()
    if noise and interventions:
        out += [
            "### Which per-class changes clear run-to-run noise?",
            "",
            "Every single-seed intervention, for the two classes the error analysis singles",
            "out, against the mean and range of the identical baseline runs (§1c):",
            "",
            "| intervention | Δ non-drivable | outside range? | Δ living-thing | outside range? |",
            "|---|---|---|---|---|",
        ]
        verdicts: dict[str, dict[str, list[str]]] = {}
        for label, row in interventions:
            cells = [label]
            for name in ("non-drivable", "living-thing"):
                key, limits = f"iou/{name}", band(noise, f"iou/{name}")
                if row.get(key) is None or not limits:
                    cells += ["—", "—"]
                    continue
                value = float(row[key])
                outside = value < limits[0] or value > limits[1]
                direction = "worse" if value < limits[0] else "better" if value > limits[1] else "noise"
                verdicts.setdefault(name, {}).setdefault(direction, []).append(label)
                cells += [f"{value - mean_of(noise, key):+.4f}", "yes" if outside else "no"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        for name in ("non-drivable", "living-thing"):
            groups = verdicts.get(name) or {}
            parts = [f"{direction} under {len(groups[direction])} ({', '.join(groups[direction])})"
                     for direction in ("worse", "better") if groups.get(direction)]
            within = len(groups.get("noise") or [])
            out.append(f"- **{name}:** " + ("; ".join(parts) if parts else "no change clears the noise")
                       + (f"; inside the noise for {within}." if within else "."))
        out.append("")
    return "\n".join(out)


def section_pretrain() -> str:
    rows = read_csv("pretrain_ablation")
    if not rows:
        return ""
    return "\n".join([
        "## 4. Encoder initialisation",
        "",
        "With only ~1400 labelled training frames, how much of the result comes from ImageNet",
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
        f"{report.get('n_fit_pixels', 0) + report.get('n_eval_pixels', 0):,} labelled pixels are",
        "drawn uniformly across every validation frame. Temperature is fitted on one half and",
        "every number below is computed on the other, which the fit never saw. The halves",
        "share frames, so this guards the one fitted parameter against its own fit, not",
        "against frame-level correlation.",
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


def section_heldout() -> str:
    """5b. A number model selection never saw."""
    val, test = read_json("heldout_val"), read_json("heldout_test")
    if not val or not test:
        return ""
    splits = (dataset_stats().get("splits") or {}).get("sequence3") or {}
    out = [
        "## 5b. A genuinely held-out test set",
        "",
        "Everywhere else in this report, validation does double duty: it is the",
        "early-stopping criterion *and* the set the number is reported on, so model",
        "selection has seen it. IDD withholds its own test labels, so a third partition is",
        "carved from the labelled frames instead — drive-disjoint from **both** train and",
        "validation, since a test set sharing drives with validation is contaminated",
        "through model selection just as surely as through training."
        + (f" That leaves {splits['n_train']} training frames." if splits.get("n_train") else ""),
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
            f"Validation minus test: **{drop:+.4f} mIoU**, from one run. It measures what early "
            "stopping bought itself on the validation set, plus the difference between two "
            "different sets of drives. It does not measure the optimism of choosing the best "
            "of many configurations on validation, which the headline numbers carry.",
            "",
        ]
    if test.get("boundary_miou") and test.get("miou"):
        out += [
            f"Boundary mIoU trails region mIoU by {float(test['miou']) - float(test['boundary_miou']):.4f} "
            "— the model places regions better than it places their edges.",
            "",
        ]
    scratch_val, scratch_test = read_json("heldout_scratch_val"), read_json("heldout_scratch_test")
    if scratch_val and scratch_test and val.get("miou") and test.get("miou"):
        gain = float(test["miou"]) - float(scratch_test["miou"])
        out += [
            "**An earlier run of this experiment trained from random initialisation**, through a",
            "command-line bug that dropped the ImageNet weights (since fixed, and pinned by",
            "`tests/test_config.py`). It is kept as what it accidentally became: a pretraining",
            f"comparison on this split. From scratch it scored {fmt(scratch_val.get('miou'))} val /",
            f"{fmt(scratch_test.get('miou'))} test, so ImageNet initialisation is worth "
            f"**{gain:+.4f} test mIoU** here (one run each).",
            "",
        ]
    return "\n".join(out)


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


def section_fgsm() -> str:
    rows = read_csv("fgsm")
    if not rows:
        return ""
    for row in rows:
        row["_budget"] = f"{fmt(row.get('epsilon_255'))}/255"
        row["_share"] = pct(float(row["retention"])) if row.get("retention") is not None else "—"
    return "\n".join([
        "## 6c. Adversarial perturbation",
        "",
        "The worst-case counterpart to the natural corruptions: a single-step FGSM",
        "perturbation, budgeted in pixel space, against the clean validation set.",
        "",
        table(rows, [("_budget", "ε"), ("miou", "mIoU"), ("_share", "share of clean mIoU")]),
        "",
    ])


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
            f"| {row['corruption']} s{fmt(row['severity'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['mean_confidence'])} | {fmt(row['overconfidence'])} | "
            f"**{fmt(row['ece_deployed_t'])}** | {fmt(row['ece_refit_t'])} |"
        )

    worst = max(plain, key=lambda r: float(r["ece_deployed_t"]))
    ratio = float(worst["ece_deployed_t"]) / max(float(clean["ece_deployed_t"]), 1e-9)
    out += [
        "",
        f"Worst case: `{worst['corruption']}` at severity {fmt(worst['severity'])}, where "
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
                f"| {row['corruption']} s{fmt(row['severity'])} | {fmt(row['ece_deployed_t'])} | "
                f"{fmt(after['ece_deployed_t'])} | {fmt(row['accuracy'])} | "
                f"{fmt(after['accuracy'])} |"
            )

    out += [
        "",
        "### Selective prediction",
        "",
        "If the confidence number carries information, abstaining on the least confident",
        "pixels should raise accuracy on the rest. This is the operating-point question a",
        "perception stack actually asks. (The serving layer still uses a fixed 0.6 for its",
        "low-confidence share; the threshold measured here is not wired into it.)",
        "",
        "| condition | coverage at 95% accuracy | risk-coverage AUC |",
        "|---|---|---|",
        f"| clean | {fmt(clean.get('coverage_at_acc_0.95'), 3)} | "
        f"{fmt(clean.get('risk_coverage_auc'))} |",
    ]
    for row in plain:
        out.append(
            f"| {row['corruption']} s{fmt(row['severity'])} | "
            f"{fmt(row.get('coverage_at_acc_0.95'), 3)} | {fmt(row.get('risk_coverage_auc'))} |"
        )
    out += ["", "![calibration under shift](figures/shift_calibration.png)", ""]
    return "\n".join(out)


def section_stability() -> str:
    rows = read_csv("stability")
    if not rows:
        return ""
    gaps = dataset_stats().get("frame_gaps") or {}
    measured = (
        f" (median gap {fmt(float(gaps['median_gap']), 0)} frame indices, only "
        f"{pct(gaps['share_within_30'])} within 30 frames)"
        if gaps.get("median_gap") and gaps.get("share_within_30") is not None else ""
    )
    return "\n".join([
        "## 7. Prediction stability",
        "",
        "A deployed segmenter sees near-identical scenes repeatedly; one that flips classes",
        "under imperceptible input changes is unusable at any mIoU. The natural measurement is",
        "frame-to-frame flicker on video — **which IDD Lite cannot support**: frames within a",
        f"drive are not adjacent{measured}, so a flicker number computed from them would",
        "really be measuring scene change.",
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
    for row in rows:
        row["_miou"] = fmt(row.get("miou"))
    out = [
        "## 8. Compression and deployment",
        "",
        "Exported to ONNX and quantized to INT8 with static, calibrated post-training",
        "quantization. Both precisions are benchmarked through ONNX Runtime **on CPU with a",
        "pinned thread count**: Apple silicon has no INT8 path through MPS, so a GPU",
        "comparison would silently run the quantized model in FP32 and report a meaningless",
        "speedup.",
        "",
        table(rows, [("precision", "precision"), ("_miou", "mIoU"), ("latency_ms_mean", "latency ms"),
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
    teacher = next((r for r in rows if r.get("variant") == "teacher"), None)
    noise = band(baseline_group(read_csv("runs")) or [], "miou")
    out = [
        "## 8b. Knowledge distillation",
        "",
        "The comparison that matters is not whether a distilled student beats its teacher —",
        "it will not — but whether it beats **the same student trained alone**, on one",
        "explicit recipe shared by both arms (`compression.distill.comparison_recipe`). Only",
        "that isolates the value of the teacher's soft targets from the value of simply",
        "having a smaller model."
        + (f" The teacher is `{teacher.get('checkpoint')}`." if teacher and teacher.get("checkpoint") else ""),
        "",
        table(rows, [("variant", "variant"), ("architecture", "architecture"),
                     ("encoder", "encoder"), ("batch_size", "batch"), ("miou", "mIoU"),
                     ("params", "params"), ("gain_over_alone", "vs student alone"),
                     ("gap_to_teacher", "vs teacher")]),
        "",
    ]
    if distilled and distilled.get("gain_over_alone") is not None:
        gain = float(distilled["gain_over_alone"])
        spread = noise[1] - noise[0] if noise else None
        if spread is not None and abs(gain) <= spread:
            verdict = (f"**Unresolved: {gain:+.4f} mIoU against the student trained alone**, "
                       f"inside the {spread:.4f} spread of identical runs (§1c). One seed per "
                       "arm cannot say whether distillation helps here.")
        else:
            verdict = (f"**Distillation {'helped' if gain > 0 else 'did not help'}: {gain:+.4f} "
                       "mIoU against the student trained alone**"
                       + (f", outside the {spread:.4f} spread of identical runs (§1c), at one "
                          "seed per arm." if spread is not None else ", at one seed per arm."))
        out += [verdict, ""]
    return "\n".join(out)


def section_pareto() -> str:
    rows = read_csv("compression_pareto")
    if not rows:
        return ""
    from compression.pareto import pareto_frontier

    frontier = {(r["variant"], r["precision"]) for r in pareto_frontier(rows)}
    for row in rows:
        row["_pareto"] = "yes" if (row["variant"], row["precision"]) in frontier else ""
        row["_miou"] = fmt(row.get("miou"))
    noise = band(baseline_group(read_csv("runs")) or [], "miou")
    out = [
        "## 8c. Deployment Pareto",
        "",
        "Every variant at both precisions, benchmarked identically. A configuration is",
        "*dominated* when another is at least as fast and at least as accurate; only the",
        "survivors represent a real deployment choice.",
        "",
        table(rows, [("variant", "variant"), ("precision", "precision"), ("_miou", "mIoU"),
                     ("latency_ms_mean", "latency ms"), ("latency_ms_p95", "p95 ms"),
                     ("size_mb", "size MB"), ("_pareto", "Pareto-optimal")], 2),
        "",
    ]
    if noise:
        out += [
            f"Dominance between variants whose mIoU differs by less than {noise[1] - noise[0]:.4f}",
            "— the spread of identical runs (§1c) — is decided by noise, not by the variants.",
            "",
        ]
    out += ["![compression Pareto](figures/compression_pareto.png)", ""]
    return "\n".join(out)


def section_caveat() -> str:
    """8d. Does the road-edge caveat fire when the score is actually optimistic?"""
    report = read_json("road_edge_caveat")
    if not report:
        return ""
    rules = report.get("rules") or {}
    labels = {
        "shipped": "road edge in or near the corridor (shipped)",
        "previous": "predicted non-drivable in the corridor (replaced)",
    }
    out = [
        "## 8d. Does the road-edge caveat fire when it should?",
        "",
        "The drivable-path score counts every corridor pixel predicted drivable, so wherever",
        "true non-drivable is absorbed into drivable *inside the corridor*, the score is",
        "optimistic. Measured with the served model and the API's own code on "
        f"{report.get('n_frames')} validation frames: {report.get('events_any')} frames have",
        "some in-corridor absorption, and "
        f"{report.get('events_1pct')} lose at least 1% of the corridor to it.",
        "",
        "| rule | frames caught | caught (≥1% of corridor) | absorbed pixels in flagged frames | fires on |",
        "|---|---|---|---|---|",
    ]
    for key in ("shipped", "previous"):
        rule = rules.get(key)
        if not rule:
            continue
        recall = rule.get("absorbed_pixel_recall")
        out.append(
            f"| {labels[key]} | {rule['frames_caught_any']}/{report.get('events_any')} | "
            f"{rule['frames_caught_1pct']}/{report.get('events_1pct')} | "
            f"{pct(recall) if recall is not None else '—'} | {pct(rule['firing_rate'], 0)} of frames |"
        )
    out += [
        "",
        "The replaced rule needed *predicted* non-drivable in the corridor, so it was silent",
        "exactly when absorption was total. The shipped rule keys on the road edge the model",
        "draws, which absorption cannot remove, and pays for that in false alarms.",
        "",
        "The margin was chosen on these same frames (sweep below), so the shipped rule's recall",
        "is optimistic by construction:",
        "",
        "| margin | frames caught | absorbed pixels in flagged frames | fires on |",
        "|---|---|---|---|",
    ]
    for row in report.get("margin_sweep") or []:
        recall = row.get("absorbed_pixel_recall")
        out.append(f"| {row['margin_px']} px | {row['frames_caught_any']}/{report.get('events_any')} | "
                   f"{pct(recall) if recall is not None else '—'} | {pct(row['firing_rate'], 0)} |")
    out.append("")
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
        shares = dict(zip(names, geometry["pixel_fraction"], strict=False))
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
            out.append(f"| {name} | {pct(frequency)} | {pct(boundary)} | {fmt(class_iou)} |")
        r_boundary = correlation["r_iou_vs_boundary_fraction"]
        r_frequency = correlation["r_iou_vs_log_frequency"]
        decisive = ""
        if shares.get("barrier-structures") and shares.get("living-thing"):
            ratio = shares["barrier-structures"] / shares["living-thing"]
            decisive = (" There is a decisive single case: `barrier-structures` is "
                        f"{ratio:.1f}× more common than `living-thing` and scores about the same "
                        "IoU, which a rarity story cannot account for.")
        stride_done = len(read_csv("stride_ablation")) >= 2
        out += [
            "",
            f"Across the {correlation['n_classes']} classes, IoU correlates at "
            f"**r = {r_boundary:+.2f}** (r² = {r_boundary ** 2:.2f}) with boundary share, against "
            f"**r = {r_frequency:+.2f}** (r² = {r_frequency ** 2:.2f}) with log frequency. Geometry "
            "explains the spread better than rarity does." + decisive,
            "",
            "**Stated with its limits.** The two predictors are themselves correlated at "
            f"r = {correlation['r_boundary_vs_log_frequency']:+.2f} — rare classes here *are* the "
            f"thin ones — and {correlation['n_classes']} points cannot separate them. "
            + ("§3b tests it directly by halving the encoder stride, and marks which per-class "
               "changes clear run-to-run noise."
               if stride_done else
               "The experiment that would settle it is one arm at `encoder_output_stride=8`, "
               "everything else fixed: if geometry is binding, the thin classes gain "
               "disproportionately."),
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
    dirty = sum(truthy(r.get("git_dirty")) for r in runs)
    stats = dataset_stats()
    data_line = (
        f"IDD Lite (`idd20k_lite`) — {stats['n_frames']} labelled frames across "
        f"{stats['n_drives']} drive sequences,"
        if stats.get("n_frames") else "IDD Lite (`idd20k_lite`),"
    )
    header = [
        "# Results",
        "",
        "*Generated by `scripts/build_results.py` from the CSVs and JSON reports in `results/`.",
        "Do not edit by hand.*",
        "",
        "**Task.** 7-class semantic segmentation of unstructured Indian road scenes, using IDD's",
        "own level-1 label hierarchy: `drivable`, `non-drivable`, `living-thing`, `vehicles`,",
        "`barrier-structures`, `construction-vegetation`, `sky`.",
        "",
        f"**Data.** {data_line} at 320×227, trained at 320×224. Every dataset in this project",
        "comes from IDD; there are no external data sources and no synthetic labels.",
        "",
        "**Scale, stated plainly.** All results are at IDD Lite scale on a single M1 Pro (MPS).",
        "This is a small dataset and the absolute mIoU values reflect that. The comparisons are",
        "controlled — one factor varies at a time, with everything else held fixed — so the",
        "*differences* between rows carry the meaning, within the noise §1c measures.",
        "",
        f"Runs recorded: {len(runs)}, {dirty} of them with uncommitted changes at run time (their",
        f"git SHA does not fully identify the code that ran). Report built at `{git_sha()}`.",
        "",
        "---",
        "",
    ]

    sections = [
        section_split(), section_seed_variance(), section_noise_floor(), section_loss(),
        section_arch(), section_stride(), section_pretrain(),
        section_calibration(), section_heldout(),
        section_robustness(), section_shift_calibration(), section_fgsm(), section_stability(),
        section_compression(), section_distillation(), section_pareto(), section_caveat(),
        section_errors(),
    ]
    body = [s for s in sections if s]
    if not body:
        body = ["_No experiment outputs found yet. Run `python scripts/run_ablations.py --suite split`._", ""]

    text = "\n".join(header) + "\n---\n\n".join(body)
    text += ("\n---\n\n## Reproducing\n\n```bash\n"
             "./scripts/reproduce.sh   # builds the data, runs every suite, then every analysis\n"
             "```\n")

    out_path = RESULTS / "RESULTS.md"
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text.splitlines())} lines, {len(body)} sections)")


if __name__ == "__main__":
    main()
