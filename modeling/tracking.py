"""Lightweight experiment tracking: one CSV row per run, plus TensorBoard scalars.

Deliberately not MLflow. A single append-only CSV that records the full config, the seed,
and the git SHA is enough to reconstruct any number in ``RESULTS.md``, and it costs no
setup for a reader trying to reproduce the work.
"""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modeling.config import REPO_ROOT

#: Paths the pipeline itself writes, or that cannot change what a run computes. Excluded
#: from the dirty check: every run rewrites ``results/runs.csv``, a tracked file, so
#: counting ``results/`` would mark every run after the first in a session as dirty,
#: whatever the state of the code.
NOT_CODE = (":(exclude)results", ":(exclude)docs", ":(exclude,glob)**/*.md")


def git_sha(short: bool = True, root: str | Path = REPO_ROOT) -> str:
    """Current commit SHA, or ``"nogit"`` outside a repository.

    Recorded per run so a results row can always be tied back to the exact code that
    produced it.
    """
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=5, cwd=root)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def git_dirty(root: str | Path = REPO_ROOT) -> bool:
    """Whether the *code* has uncommitted changes at run time.

    Only code counts: results and documentation are excluded (see :data:`NOT_CODE`), so
    ``False`` means the recorded SHA identifies exactly the code that ran.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", *NOT_CODE],
            capture_output=True, text=True, timeout=5, cwd=root,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on a sidecar of ``path`` for a read-modify-write.

    Two training processes -- one per GPU -- each append to the same run log, and an
    append here rewrites the whole file, so without the lock the later writer silently
    drops the other's row. POSIX ``flock`` (macOS, Linux); a no-op where unavailable,
    which leaves single-process use exactly as safe as before.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with path.with_name(path.name + ".lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def log_run(row: dict[str, Any], path: str | Path = "results/runs.csv") -> Path:
    """Append one run to the experiment CSV, widening the header if new keys appear.

    Safe to call from several processes at once: the read-modify-write holds a file lock,
    and the new file replaces the old one atomically, so a crash mid-write cannot leave a
    truncated log either.

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

    with _exclusive(path):
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

        staging = path.with_name(path.name + ".tmp")
        with staging.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for old in existing:
                writer.writerow(old)
            writer.writerow(row)
        os.replace(staging, path)
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
