"""Lightweight experiment tracking: one CSV row per run, plus TensorBoard scalars.

Deliberately not MLflow. A single append-only CSV that records the full config, the seed,
and the git SHA is enough to reconstruct any number in ``RESULTS.md``, and it costs no
setup for a reader trying to reproduce the work.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_sha(short: bool = True) -> str:
    """Current commit SHA, or ``"nogit"`` outside a repository.

    Recorded per run so a results row can always be tied back to the exact code that
    produced it.
    """
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def git_dirty() -> bool:
    """Whether the working tree has uncommitted changes at run time."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def log_run(row: dict[str, Any], path: str | Path = "results/runs.csv") -> Path:
    """Append one run to the experiment CSV, widening the header if new keys appear.

    Args:
        row: Flat mapping of column name to value.
        path: Destination CSV.

    Returns:
        The CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        **row,
    }

    existing: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
            fieldnames = list(reader.fieldnames or [])

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for old in existing:
            writer.writerow(old)
        writer.writerow(row)
    return path


class TensorBoard:
    """Thin optional wrapper around ``torch.utils.tensorboard.SummaryWriter``.

    Degrades to a no-op when TensorBoard is unavailable, so training never fails because
    a logging dependency is missing.
    """

    def __init__(self, log_dir: str | Path | None):
        self.writer = None
        if log_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(log_dir))
        except Exception:  # noqa: BLE001 - logging must never break training
            self.writer = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def scalars(self, values: dict[str, float], step: int) -> None:
        for tag, value in values.items():
            self.scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


__all__ = ["log_run", "git_sha", "git_dirty", "TensorBoard"]
