"""Distillation-loss tests.

The T-squared correction and the ignore-pixel masking are both easy to omit and neither
fails loudly, so they are pinned here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from compression.distill import DistillationLoss

IGNORE = 255


@pytest.fixture
def batch():
    torch.manual_seed(0)
    n, c, h, w = 2, 7, 8, 8
    student = torch.randn(n, c, h, w)
    teacher = torch.randn(n, c, h, w)
    target = torch.randint(0, c, (n, h, w))
    target[:, :2, :] = IGNORE
    return student, teacher, target


def test_alpha_zero_reduces_to_cross_entropy(batch):
    student, teacher, target = batch
    expected = F.cross_entropy(student, target.long(), ignore_index=IGNORE)
    got = DistillationLoss(alpha=0.0)(student, teacher, target)
    assert torch.allclose(got, expected)


def test_student_matching_teacher_has_no_soft_term(batch):
    _, teacher, target = batch
    loss = DistillationLoss(alpha=1.0)(teacher.clone(), teacher, target)
    assert abs(float(loss)) < 1e-5


def test_temperature_squared_keeps_gradient_scale_stable(batch):
    """Without the T^2 factor the soft-target gradient would shrink as 1/T^2, so retuning
    T would silently retune the effective learning rate on the distillation signal."""
    student, teacher, target = batch
    norms = []
    for temperature in (1.0, 2.0, 4.0, 8.0):
        s = student.clone().requires_grad_(True)
        DistillationLoss(alpha=1.0, temperature=temperature)(s, teacher, target).backward()
        norms.append(float(s.grad.norm()))
    # Across a 8x temperature range the gradient norm should stay within a small band.
    assert max(norms) / min(norms) < 1.5, f"gradient scale drifted with T: {norms}"


def test_ignored_pixels_get_no_gradient(batch):
    """Unlabelled regions are ambiguous, so the teacher's output there is its least
    trustworthy signal and must not train the student."""
    student, teacher, target = batch
    s = student.clone().requires_grad_(True)
    DistillationLoss(alpha=1.0)(s, teacher, target).backward()
    assert float(s.grad[:, :, :2, :].abs().sum()) == 0.0
    assert float(s.grad[:, :, 2:, :].abs().sum()) > 0.0


def test_loss_is_finite_and_differentiable(batch):
    student, teacher, target = batch
    s = student.clone().requires_grad_(True)
    loss = DistillationLoss(alpha=0.5, temperature=4.0)(s, teacher, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(s.grad).all()


def test_all_pixels_ignored_does_not_produce_nan():
    """A batch with no labelled pixels must not divide by zero."""
    torch.manual_seed(0)
    student = torch.randn(1, 3, 4, 4, requires_grad=True)
    teacher = torch.randn(1, 3, 4, 4)
    target = torch.full((1, 4, 4), IGNORE, dtype=torch.long)
    loss = DistillationLoss(alpha=1.0)(student, teacher, target)
    assert torch.isfinite(loss)


def test_pareto_frontier_excludes_dominated_configurations():
    """A configuration is dominated when another is at least as fast and as accurate;
    only the survivors represent a real deployment choice."""
    from compression.pareto import pareto_frontier

    rows = [
        {"variant": "teacher", "precision": "fp32", "latency_ms_mean": 100, "miou": 0.70},
        {"variant": "teacher", "precision": "int8", "latency_ms_mean": 60, "miou": 0.69},
        {"variant": "student", "precision": "fp32", "latency_ms_mean": 40, "miou": 0.65},
        {"variant": "student", "precision": "int8", "latency_ms_mean": 25, "miou": 0.64},
        {"variant": "dominated", "precision": "fp32", "latency_ms_mean": 80, "miou": 0.60},
    ]
    frontier = pareto_frontier(rows)
    assert all(r["variant"] != "dominated" for r in frontier)
    assert len(frontier) == 4
    # Sorted by latency, and accuracy must increase along it.
    latencies = [r["latency_ms_mean"] for r in frontier]
    mious = [r["miou"] for r in frontier]
    assert latencies == sorted(latencies)
    assert mious == sorted(mious)


def test_pareto_frontier_keeps_a_single_point():
    from compression.pareto import pareto_frontier

    rows = [{"variant": "only", "precision": "fp32", "latency_ms_mean": 10, "miou": 0.5}]
    assert pareto_frontier(rows) == rows
