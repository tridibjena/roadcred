"""Loss tests, including the identities that define the ablation arms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modeling.losses import (
    BoundaryWeightedCE,
    CrossEntropy,
    DiceLoss,
    TverskyLoss,
    compute_class_weights,
    make_loss,
)


@pytest.fixture
def banded():
    """Structured 3-class target with real boundaries, and matching logits."""
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 5:10, :] = 1
    target[:, 10:, :] = 2
    torch.manual_seed(0)
    return torch.randn(1, 3, 16, 16), target


def test_dice_is_tversky_at_half_half(banded):
    logits, target = banded
    assert torch.allclose(DiceLoss()(logits, target), TverskyLoss(0.5, 0.5)(logits, target))


def test_tversky_matches_hand_computation():
    """Target [0,1,1,1], hard prediction [0,0,1,1]: class1 tp=2 fn=1, class0 tp=1 fp=1."""
    logits = torch.tensor([[[[20.0, 20.0, 0.0, 0.0]], [[0.0, 0.0, 20.0, 20.0]]]])
    target = torch.tensor([[[0, 1, 1, 1]]])
    for alpha, beta in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]:
        expected = 1 - ((1 / (1 + alpha)) + (2 / (2 + beta))) / 2
        got = TverskyLoss(alpha=alpha, beta=beta, eps=0.0)(logits, target).item()
        assert abs(got - expected) < 1e-5, f"alpha={alpha} beta={beta}"


def test_perfect_prediction_gives_zero_cross_entropy(banded):
    _, target = banded
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    assert CrossEntropy()(perfect, target).item() < 1e-5


def test_ignore_index_excluded_from_loss():
    """Pixels marked 255 must not influence the loss at all."""
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    ignored = target.clone()
    ignored[:, 2:, :] = 255
    # Changing predictions only under the ignored region must not change the loss.
    baseline = CrossEntropy()(logits, ignored)
    perturbed = logits.clone()
    perturbed[:, :, 2:, :] += 10.0
    assert torch.allclose(baseline, CrossEntropy()(perturbed, ignored))


def test_class_weights_are_normalised_and_inverse_to_frequency():
    counts = np.array([100000, 2000, 1300, 8000, 11000, 26000, 18000])
    weights = compute_class_weights(counts)
    assert abs(float(weights.mean()) - 1.0) < 1e-5
    # The rarest class must receive the largest weight.
    assert int(weights.argmax()) == int(np.argmin(counts))
    assert int(weights.argmin()) == int(np.argmax(counts))


def test_absent_class_gets_zero_weight_not_infinite():
    weights = compute_class_weights(np.array([100, 0, 50]))
    assert float(weights[1]) == 0.0
    assert torch.isfinite(weights).all()


def test_boundary_band_finds_real_boundaries(banded):
    """Bands must sit at the class transitions, not everywhere and not nowhere."""
    _, target = banded
    band = BoundaryWeightedCE()._boundary_band(target)
    fraction = float(band.float().mean())
    assert 0.0 < fraction < 0.6
    # Rows adjacent to the transitions are flagged; the interior of a band is not.
    assert band[0, 4, 8] and band[0, 5, 8]
    assert not band[0, 0, 8]


def test_boundary_weighting_changes_the_loss(banded):
    logits, target = banded
    plain = CrossEntropy()(logits, target)
    weighted = BoundaryWeightedCE(boundary_weight=5.0)(logits, target)
    assert not torch.allclose(plain, weighted)


@pytest.mark.parametrize("name", ["ce", "weighted_ce", "dice", "tversky", "boundary"])
def test_every_ablation_arm_is_finite_and_differentiable(name, banded):
    logits, target = banded
    logits = logits.clone().requires_grad_(True)
    weights = compute_class_weights(np.array([50, 30, 20]))
    loss = make_loss(name, class_weights=weights)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_weighted_ce_requires_weights(banded):
    with pytest.raises(ValueError, match="requires class_weights"):
        make_loss("weighted_ce")


@pytest.mark.parametrize("name", ["ce", "weighted_ce", "dice", "tversky", "boundary"])
def test_all_ignored_batch_does_not_produce_nan(name):
    """Regression: F.cross_entropy's mean reduction returns NaN when every pixel is
    ignored, and `0 * NaN` is still NaN, so even a zero blend weight does not rescue a
    combined loss. A single NaN batch propagates through AdamW and destroys the weights
    permanently, so this must be a hard zero rather than merely 'rare in practice'."""
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), 255, dtype=torch.long)
    weights = compute_class_weights(np.array([50, 30, 20]))

    loss = make_loss(name, class_weights=weights)(logits, target)
    assert torch.isfinite(loss), f"{name} produced {loss} on an all-ignored batch"
    assert float(loss) == pytest.approx(0.0, abs=1e-6)

    loss.backward()
    assert torch.isfinite(logits.grad).all(), f"{name} produced non-finite gradients"


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        make_loss("nonexistent")
