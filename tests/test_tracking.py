"""Run-log tests: concurrent appends must not lose rows, and "dirty" must mean the code."""

from __future__ import annotations

import csv
import multiprocessing
import subprocess
from pathlib import Path

import pytest

from modeling.tracking import git_dirty, log_run


def _append_rows(path: str, worker: int, count: int) -> None:
    for index in range(count):
        log_run({"worker": worker, "index": index}, path)


def test_concurrent_appends_lose_no_rows(tmp_path):
    """Regression guard for running one training process per GPU: an append rewrites the
    whole file, so without the lock the later writer silently drops the other's rows."""
    path = tmp_path / "runs.csv"
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_append_rows, args=(str(path), w, 8)) for w in range(4)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(timeout=60)
        assert process.exitcode == 0
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert {(r["worker"], r["index"]) for r in rows} == {
        (str(w), str(i)) for w in range(4) for i in range(8)
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "train.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("# notes\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "runs.csv").write_text("miou\n0.5\n")
    git("add", ".")
    git("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return tmp_path


def test_outputs_and_docs_do_not_make_the_code_dirty(repo):
    """Every run rewrites results/runs.csv, a tracked file. Counting it would mark every
    run after the first in a session as dirty whatever the state of the code."""
    (repo / "results" / "runs.csv").write_text("miou\n0.5\n0.6\n")
    (repo / "README.md").write_text("# edited notes\n")
    assert git_dirty(repo) is False


def test_a_code_change_is_dirty(repo):
    (repo / "train.py").write_text("x = 2\n")
    assert git_dirty(repo) is True
