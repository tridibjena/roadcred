"""Hand-written PyTorch training loop for semantic segmentation.

Everything the ablations vary -- loss family, architecture, encoder init, split mode,
seed -- runs through this one loop, so any difference in the results tables is
attributable to the thing that was varied and not to a difference in training recipe.

Run::

    python -m modeling.train --data data/processed/level1_official --loss ce --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import ConfusionMatrix
from modeling.architectures import build_model, count_parameters
from modeling.config import resolve_device, set_seed
from modeling.dataset import build_loaders, training_class_counts
from modeling.losses import compute_class_weights, make_loss
from modeling.tracking import TensorBoard, log_run


def cosine_schedule(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_factor: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, stepped per optimiser step.

    Warmup matters here because the decoder is randomly initialised while the encoder is
    pretrained; without it the first few large-gradient steps damage the encoder features
    before the decoder produces anything meaningful.
    """

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    class_names: list[str] | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Run validation and return the full metric summary.

    Metrics come from a single confusion matrix accumulated over the whole split, so a
    rare class appearing in only a few frames is weighted correctly rather than being
    averaged per batch.
    """
    model.eval()
    confusion = ConfusionMatrix(n_classes)
    total_loss, n_batches = 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, targets).detach().item()
            n_batches += 1
        confusion.update(logits, targets)

    summary = confusion.summary(class_names)
    if n_batches:
        summary["val_loss"] = total_loss / n_batches
    summary["_confusion"] = confusion.matrix  # type: ignore[assignment]
    return summary


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    grad_clip: float = 1.0,
    log: TensorBoard | None = None,
    global_step: int = 0,
) -> tuple[float, int]:
    """Train for one epoch. Returns ``(mean_loss, new_global_step)``."""
    model.train()
    running, count = 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running += loss.detach().item()
        count += 1
        global_step += 1
        if log is not None and global_step % 20 == 0:
            log.scalar("train/loss_step", loss.detach().item(), global_step)
            log.scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    return (running / max(1, count)), global_step


def train(
    data_root: str | Path,
    *,
    architecture: str = "deeplabv3plus",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    loss: str = "ce",
    epochs: int = 40,
    batch_size: int = 16,
    imgsz: tuple[int, int] = (224, 320),
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "auto",
    workers: int = 0,
    patience: int = 12,
    warmup_frac: float = 0.05,
    grad_clip: float = 1.0,
    class_names: list[str] | None = None,
    out_dir: str | Path = "checkpoints",
    run_name: str | None = None,
    tensorboard: bool = True,
    results_csv: str | Path = "results/runs.csv",
    tversky_beta: float = 0.7,
    loss_alpha: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model and return its best validation metrics.

    Args:
        data_root: Prepared variant directory, e.g. ``data/processed/level1_official``.
        architecture: Key from :data:`modeling.architectures.SMP_ARCHITECTURES`.
        encoder: Backbone name.
        encoder_weights: ``imagenet`` or ``None``.
        loss: Loss key for :func:`modeling.losses.make_loss`.
        epochs: Maximum epochs; early stopping may end training sooner.
        batch_size: Samples per step.
        imgsz: ``(height, width)``, both multiples of 32.
        lr: Peak learning rate after warmup.
        weight_decay: AdamW decay.
        seed: Seeds model init, data order, and augmentation.
        device: ``auto`` / ``mps`` / ``cuda`` / ``cpu``.
        workers: Dataloader workers. Defaults to 0: augmented loading measures ~640
            img/s single-process against ~37 img/s of GPU throughput, so worker
            processes add spawn overhead and failure modes for no gain.
        patience: Stop after this many epochs without a new best val mIoU.
        warmup_frac: Fraction of total steps spent warming up.
        grad_clip: Max gradient norm; 0 disables.
        class_names: Names for per-class metric keys.
        out_dir: Where the best checkpoint is written.
        run_name: Identifier used for the checkpoint, TensorBoard dir, and CSV row.
        tensorboard: Whether to write TensorBoard scalars.
        results_csv: Experiment log to append to.
        tversky_beta: False-negative weight for the Tversky arm.
        loss_alpha: Blend weight for combined region losses.
        verbose: Print per-epoch progress.

    Returns:
        Best-epoch metrics plus timing, parameter count, and the confusion matrix.
    """
    device_t = resolve_device(device)
    set_seed(seed)
    data_root = Path(data_root)
    class_names = class_names or _default_class_names(data_root)
    n_classes = len(class_names)
    run_name = run_name or f"{architecture}_{encoder}_{loss}_s{seed}_{data_root.name}"

    train_loader, val_loader = build_loaders(data_root, imgsz, batch_size, workers, seed)

    class_weights = None
    if loss == "weighted_ce":
        counts = training_class_counts(data_root, n_classes)
        class_weights = compute_class_weights(counts).to(device_t)

    model = build_model(architecture, encoder, n_classes, encoder_weights).to(device_t)
    criterion = make_loss(
        loss, class_weights=class_weights, alpha=loss_alpha, tversky_beta=tversky_beta
    ).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * len(train_loader))
    scheduler = cosine_schedule(optimizer, total_steps, int(warmup_frac * total_steps))

    board = TensorBoard(Path("runs") / run_name if tensorboard else None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{run_name}.pt"

    best: dict[str, Any] = {"miou": -1.0}
    best_epoch, stale, global_step = -1, 0, 0
    started = time.time()

    for epoch in range(epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device_t, grad_clip, board, global_step,
        )
        metrics = evaluate(model, val_loader, device_t, n_classes, class_names, criterion)
        confusion = metrics.pop("_confusion")

        board.scalar("train/loss_epoch", train_loss, epoch)
        board.scalars({f"val/{k}": v for k, v in metrics.items() if not math_isnan(v)}, epoch)

        if metrics["miou"] > best["miou"]:
            best = dict(metrics)
            best["confusion"] = confusion
            best_epoch, stale = epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "encoder": encoder,
                    "n_classes": n_classes,
                    "class_names": class_names,
                    "imgsz": imgsz,
                    "epoch": epoch,
                    "miou": metrics["miou"],
                },
                checkpoint_path,
            )
        else:
            stale += 1

        if verbose:
            print(
                f"  epoch {epoch + 1:3d}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={metrics.get('val_loss', float('nan')):.4f}  "
                f"mIoU={metrics['miou']:.4f}  best={best['miou']:.4f}"
                + ("  *" if stale == 0 else ""),
                flush=True,
            )
        if patience and stale >= patience:
            if verbose:
                print(f"  early stop: no improvement for {patience} epochs", flush=True)
            break

    board.close()
    elapsed = time.time() - started

    result = {k: v for k, v in best.items() if k != "confusion"}
    result.update(
        {
            "run_name": run_name,
            "data": data_root.name,
            "architecture": architecture,
            "encoder": encoder,
            "encoder_weights": str(encoder_weights),
            "loss": loss,
            "seed": seed,
            "epochs_ran": best_epoch + 1,
            "epochs_max": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "imgsz": f"{imgsz[0]}x{imgsz[1]}",
            "params": count_parameters(model),
            "device": str(device_t),
            "train_seconds": round(elapsed, 1),
            "checkpoint": str(checkpoint_path),
        }
    )
    log_run(result, results_csv)
    (out_dir / f"{run_name}_confusion.json").write_text(
        json.dumps({"class_names": class_names, "matrix": best["confusion"].tolist()}, indent=2)
    )
    return result


def math_isnan(value: Any) -> bool:
    """True when ``value`` is a float NaN (per-class IoU is NaN for absent classes)."""
    return isinstance(value, float) and math.isnan(value)


def _default_class_names(data_root: Path) -> list[str]:
    """Read class names from the variant's dataset YAML, falling back to level-1 names."""
    import yaml

    candidate = Path("configs") / f"{data_root.name}.yaml"
    if candidate.exists():
        spec = yaml.safe_load(candidate.read_text())
        if spec and "names" in spec:
            return list(spec["names"])
    from data.label_utils import LEVEL1_NAMES

    return list(LEVEL1_NAMES)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config supplying defaults; any flag given explicitly overrides it",
    )
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--arch", default="deeplabv3plus")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", default="224x320")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    # Config supplies defaults; anything the user typed on the command line wins. Only
    # flags actually present in argv override, so a config value is never silently
    # replaced by an argparse default the user never asked for.
    from dataclasses import asdict

    from modeling.config import load_config

    settings = asdict(load_config(args.config).train) if args.config else {}
    supplied = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    cli = {
        "architecture": args.arch,
        "encoder": args.encoder,
        "encoder_weights": args.encoder_weights,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
    }
    alias = {"architecture": "arch", "batch_size": "batch_size"}
    for key, value in cli.items():
        if not settings or alias.get(key, key) in supplied:
            settings[key] = value

    imgsz = settings.get("imgsz", (224, 320))
    if isinstance(imgsz, str):
        imgsz = tuple(int(v) for v in imgsz.lower().split("x"))
    settings["imgsz"] = tuple(imgsz)

    weights = settings.get("encoder_weights")
    if isinstance(weights, str) and weights.lower() in {"none", "null", ""}:
        weights = None
    settings["encoder_weights"] = weights

    result = train(
        args.data,
        run_name=args.run_name,
        tensorboard=not args.no_tensorboard,
        **settings,
    )
    print(json.dumps({k: v for k, v in result.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
