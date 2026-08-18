"""Shared fixtures. Tests run without any dataset present unless marked otherwise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
IDD_ROOT = REPO_ROOT / "data" / "raw" / "idd_seg"


@pytest.fixture(scope="session")
def idd_available() -> bool:
    """Whether the real IDD download is present."""
    return (IDD_ROOT / "idd20k_lite").exists() or (IDD_ROOT / "leftImg8bit").exists()


@pytest.fixture
def synthetic_mask() -> np.ndarray:
    """A small deterministic label map with three horizontal bands and an ignore strip."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:10, :] = 1
    mask[10:, :] = 2
    mask[:2, :] = 255
    return mask
