"""Knowledge distillation from a larger teacher into a deployable student.

The comparison that matters is not "does distillation beat the teacher" — it will not —
but **does a distilled student beat the same student trained alone**, at identical
capacity, data, schedule and seed. Only that isolates the contribution of the teacher's
soft targets from the contribution of simply having a smaller model.

The loss blends two terms:

* **Cross-entropy against ground truth**, exactly as in normal training.
* **KL divergence against the teacher's temperature-softened distribution.** The soft
  targets carry the teacher's uncertainty between classes — that `barrier-structures` is
  often 60/40 with `construction-vegetation` — which a hard label discards entirely.

The KL term is scaled by ``T^2``. Softening by ``T`` shrinks the gradients of that term by
``1/T^2``, so without the correction the effective learning rate on the distillation
signal would change every time ``T`` is retuned, confounding the two.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.architectures import build_model, count_parameters
from modeling.config import resolve_device, set_seed
from modeling.dataset import build_loaders
from modeling.losses import IGNORE_INDEX
from modeling.tracking import TensorBoard, log_run
from modeling.train import cosine_schedule, evaluate


class DistillationLoss(nn.Module):
    """``alpha * KL(student || teacher) * T^2 + (1 - alpha) * CE(student, target)``.

    Args:
        alpha: Weight on the teacher term. 0 recovers plain training.
        temperature: Softening applied to both teacher and student logits.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        valid_pixels = target != IGNORE_INDEX
        if not valid_pixels.any():
            # See modeling.losses.CrossEntropy: an all-ignored batch makes the mean
            # reduction return NaN, which survives multiplication by (1 - alpha).
            return student_logits.sum() * 0.0
        hard = F.cross_entropy(student_logits, target.long(), ignore_index=IGNORE_INDEX)

        # Ignored pixels have no ground truth, but the teacher still has an opinion there.
        # They are excluded anyway: those regions are unlabelled precisely because they are
        # ambiguous, so the teacher's output on them is the least trustworthy signal it has.
        valid = (target != IGNORE_INDEX).unsqueeze(1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        per_pixel = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=1, keepdim=True)
        soft = (per_pixel * valid).sum() / valid.sum().clamp_min(1)

        # T^2 keeps the soft-target gradient magnitude independent of the temperature.
        return self.alpha * soft * (self.temperature**2) + (1 - self.alpha) * hard


def distill(
    data_root: str | Path,
    teacher_checkpoint: str | Path,
    *,
    architecture: str = "deeplabv3plus",
    encoder: str = "resnet18",
    encoder_weights: str | None = "imagenet",
    alpha: float = 0.5,
    temperature: float = 4.0,
    epochs: int = 30,
    batch_size: int = 8,
    imgsz: tuple[int, int] = (224, 320),
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "auto",
    patience: int = 8,
    out_dir: str | Path = "checkpoints",
    run_name: str | None = None,
    results_csv: str | Path = "results/runs.csv",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a student against a frozen teacher and return its best metrics."""
    from evaluation.eval import load_checkpoint

    device_t = resolve_device(device)
    set_seed(seed)
    data_root = Path(data_root)
    run_name = run_name or f"distill_{encoder}_a{alpha}_T{temperature}_s{seed}"

    teacher, payload = load_checkpoint(teacher_checkpoint, device_t)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    class_names = payload["class_names"]
    n_classes = len(class_names)
    train_loader, val_loader = build_loaders(data_root, imgsz, batch_size, 0, seed)

    student = build_model(architecture, encoder, n_classes, encoder_weights).to(device_t)
    criterion = DistillationLoss(alpha, temperature).to(device_t)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = cosine_schedule(optimizer, total_steps, int(0.05 * total_steps))

    board = TensorBoard(Path("runs") / run_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{run_name}.pt"

    best: dict[str, Any] = {"miou": -1.0}
    best_epoch, stale = -1, 0
    started = time.time()

    for epoch in range(epochs):
        student.train()
        running, count = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device_t)
            targets = targets.to(device_t)
            with torch.no_grad():
                teacher_logits = teacher(images)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(student(images), teacher_logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.detach().item()
            count += 1

        metrics = evaluate(student, val_loader, device_t, n_classes, class_names)
        metrics.pop("_confusion", None)
        board.scalar("train/loss_epoch", running / max(1, count), epoch)
        board.scalar("val/miou", metrics["miou"], epoch)

        if metrics["miou"] > best["miou"]:
            best, best_epoch, stale = dict(metrics), epoch, 0
            torch.save(
                {
                    "model": student.state_dict(),
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
                f"  epoch {epoch + 1:3d}/{epochs}  loss={running / max(1, count):.4f}  "
                f"mIoU={metrics['miou']:.4f}  best={best['miou']:.4f}",
                flush=True,
            )
        if patience and stale >= patience:
            break

    board.close()
    result = {
        **best,
        "run_name": run_name,
        "data": data_root.name,
        "architecture": architecture,
        "encoder": encoder,
        "loss": f"distill(a={alpha},T={temperature})",
        "teacher": str(teacher_checkpoint),
        "teacher_params": count_parameters(teacher),
        "seed": seed,
        "epochs_ran": best_epoch + 1,
        "params": count_parameters(student),
        "device": str(device_t),
        "train_seconds": round(time.time() - started, 1),
        "checkpoint": str(checkpoint_path),
    }
    log_run(result, results_csv)
    return result


def run_comparison(
    data_root: str | Path,
    teacher_checkpoint: str | Path,
    student_encoder: str = "resnet18",
    epochs: int = 30,
    seed: int = 0,
    out_csv: str | Path = "results/distillation.csv",
) -> list[dict[str, Any]]:
    """Teacher vs standalone student vs distilled student, all else held fixed."""
    import csv

    from evaluation.eval import evaluate_checkpoint
    from modeling.train import train

    rows: list[dict[str, Any]] = []

    teacher_metrics = evaluate_checkpoint(teacher_checkpoint, data_root)
    rows.append({
        "variant": "teacher",
        "encoder": teacher_metrics.get("checkpoint", ""),
        "miou": teacher_metrics["miou"],
        "params": teacher_metrics["params"],
    })
    print(f"teacher: mIoU={teacher_metrics['miou']:.4f}", flush=True)

    print("\n--- student trained alone ---", flush=True)
    baseline = train(
        data_root, encoder=student_encoder, epochs=epochs, seed=seed,
        run_name=f"student_alone_{student_encoder}_s{seed}", patience=8,
    )
    rows.append({"variant": "student_alone", "encoder": student_encoder,
                 "miou": baseline["miou"], "params": baseline["params"]})

    print("\n--- student distilled from teacher ---", flush=True)
    distilled = distill(
        data_root, teacher_checkpoint, encoder=student_encoder, epochs=epochs, seed=seed,
        run_name=f"student_distilled_{student_encoder}_s{seed}",
    )
    rows.append({"variant": "student_distilled", "encoder": student_encoder,
                 "miou": distilled["miou"], "params": distilled["params"]})

    rows[-1]["gain_over_alone"] = distilled["miou"] - baseline["miou"]
    rows[-1]["gap_to_teacher"] = distilled["miou"] - teacher_metrics["miou"]

    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "encoder", "miou", "params", "gain_over_alone", "gap_to_teacher"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="Trained teacher checkpoint")
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--student-encoder", default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/distillation.csv")
    args = parser.parse_args()

    rows = run_comparison(
        args.data, args.teacher, args.student_encoder, args.epochs, args.seed, args.out
    )
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
