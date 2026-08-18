"""Corruption and stability tests.

These guard two properties the robustness results depend on: corruptions must actually
change the image in a severity-ordered way, and the stability perturbations must be
invertible so a geometric shift is not miscounted as a prediction flip.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.corruptions import CORRUPTIONS, WEATHER_LIKE, make_corruption
from evaluation.stability import PERTURBATIONS, shift_image


@pytest.fixture
def image() -> np.ndarray:
    """A structured RGB image; random noise would hide blur and compression effects."""
    rng = np.random.default_rng(0)
    base = np.zeros((64, 96, 3), dtype=np.uint8)
    base[:32] = [120, 140, 160]
    base[32:] = [80, 70, 60]
    base[:, ::8] = [200, 200, 200]
    return np.clip(base.astype(int) + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_preserves_shape_and_dtype(name, image):
    for severity in (1, 3, 5):
        out = CORRUPTIONS[name](image, severity)
        assert out.shape == image.shape
        assert out.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_actually_changes_the_image(name, image):
    """A corruption that leaves the image untouched would silently report no degradation."""
    out = CORRUPTIONS[name](image, 5)
    assert not np.array_equal(out, image), f"{name} at severity 5 was a no-op"


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_severity_is_monotone(name, image):
    """Severity 5 must perturb at least as much as severity 1."""
    low = np.abs(CORRUPTIONS[name](image, 1).astype(int) - image.astype(int)).mean()
    high = np.abs(CORRUPTIONS[name](image, 5).astype(int) - image.astype(int)).mean()
    assert high >= low, f"{name}: severity 5 perturbs less than severity 1"


def test_make_corruption_validates_inputs():
    with pytest.raises(ValueError, match="Unknown corruption"):
        make_corruption("not_a_corruption", 3)
    with pytest.raises(ValueError, match="Severity"):
        make_corruption("fog", 9)


def test_weather_like_subset_is_valid():
    assert set(WEATHER_LIKE) <= set(CORRUPTIONS)


def test_shift_is_invertible_away_from_the_border():
    """The stability metric depends on undoing a shift exactly in the interior."""
    rng = np.random.default_rng(0)
    original = rng.integers(0, 7, (32, 40)).astype(np.uint8)
    roundtrip = shift_image(shift_image(original, 2, 2), -2, -2)
    assert np.array_equal(roundtrip[3:-3, 3:-3], original[3:-3, 3:-3])


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_perturbations_are_label_preserving_and_small(name, image):
    """Stability perturbations must be subtle -- a large change is a corruption, not noise."""
    transform, _, _ = PERTURBATIONS[name]
    out = transform(image)
    assert out.shape == image.shape and out.dtype == np.uint8
    if not name.startswith("shift"):
        difference = np.abs(out.astype(int) - image.astype(int)).mean()
        assert difference < 20, f"{name} changed the image by {difference:.1f} mean levels"
