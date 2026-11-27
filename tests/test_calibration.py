"""Calibration tests: temperature recovery and how pixels are sampled for it."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.calibration import TemperatureScaler, collect_predictions


class _Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _indexed_loader(n_images: int = 40, size: int = 16) -> DataLoader:
    """Each image's labels equal its own index, so a sample reveals which frames it saw."""
    images = torch.randn(n_images, 3, size, size)
    targets = torch.arange(n_images).view(-1, 1, 1).expand(n_images, size, size).clone()
    return DataLoader(TensorDataset(images, targets), batch_size=4, shuffle=False)


def test_sample_spans_the_whole_split_not_its_first_frames():
    """Regression: the budget used to be filled from the first batches and the loop then
    stopped, so calibration saw a prefix of the split in filename order."""
    loader = _indexed_loader(n_images=40, size=16)
    budget = 40 * 16 * 16 // 10  # a tenth of all pixels
    _, targets = collect_predictions(_Identity(), loader, torch.device("cpu"), budget, seed=0)
    assert targets.numel() <= budget
    seen = set(targets.tolist())
    assert seen == set(range(40)), "every image must contribute pixels"
    counts = torch.bincount(targets, minlength=40).float()
    assert counts.min() > 0.5 * counts.mean(), "images should contribute roughly equally"


def test_same_seed_draws_the_same_pixels():
    loader = _indexed_loader()
    a = collect_predictions(_Identity(), loader, torch.device("cpu"), 500, seed=3)[1]
    b = collect_predictions(_Identity(), loader, torch.device("cpu"), 500, seed=3)[1]
    assert torch.equal(a, b)


def test_temperature_scaler_recovers_a_known_temperature():
    """Logits sharpened by a known factor should be undone by a fitted T near that factor."""
    torch.manual_seed(0)
    n, c, true_t = 20_000, 5, 3.0
    calibrated = torch.randn(n, c)
    labels = torch.distributions.Categorical(logits=calibrated).sample()
    fitted = TemperatureScaler().fit(calibrated * true_t, labels)
    assert abs(fitted - true_t) < 0.15
