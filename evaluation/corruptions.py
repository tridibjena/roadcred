"""Synthetic test-time corruptions simulating adverse driving conditions.

These stand in for a real adverse-weather dataset. They are applied **only at test time**
-- the training augmentation pipeline deliberately excludes blur, noise and weather -- so
the resulting degradation measures genuine distribution shift rather than a train/test
augmentation mismatch.

Each corruption takes an RGB ``uint8`` image and a severity in ``1..5`` and returns an RGB
``uint8`` image. Severities are calibrated so that 1 is barely visible and 5 is severe but
still humanly interpretable; a corruption that destroys the image tests nothing useful.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _as_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def _as_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def gaussian_noise(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Sensor noise, as in a high-ISO capture."""
    scale = [0.04, 0.06, 0.10, 0.16, 0.26][severity - 1]
    rng = np.random.default_rng(severity)
    return _as_uint8(_as_float(image) + rng.normal(0, scale, image.shape))


def shot_noise(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Poisson (photon) noise, which dominates in low light."""
    rate = [60, 25, 12, 5, 3][severity - 1]
    rng = np.random.default_rng(severity)
    return _as_uint8(rng.poisson(_as_float(image) * rate) / rate)


def gaussian_blur(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Defocus / soft optics."""
    import cv2

    sigma = [0.6, 1.0, 1.6, 2.4, 3.4][severity - 1]
    return cv2.GaussianBlur(image, (0, 0), sigma)


def motion_blur(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Horizontal motion blur from vehicle speed."""
    import cv2

    length = [3, 5, 9, 13, 19][severity - 1]
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def brightness(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Over-exposure, as in direct glare."""
    delta = [0.08, 0.16, 0.25, 0.35, 0.48][severity - 1]
    return _as_uint8(_as_float(image) + delta)


def low_light(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Under-exposure with accompanying shot noise -- dusk and night driving.

    Gamma darkening alone is unrealistically clean: real low-light frames lose signal-to-
    noise as well as brightness, so noise is added in proportion to the darkening.
    """
    gain = [0.72, 0.55, 0.40, 0.28, 0.18][severity - 1]
    rng = np.random.default_rng(severity)
    darkened = _as_float(image) * gain
    return _as_uint8(darkened + rng.normal(0, 0.02 / gain, image.shape) * 0.5)


def contrast(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Haze-like contrast loss."""
    factor = [0.75, 0.6, 0.45, 0.3, 0.2][severity - 1]
    grey = _as_float(image).mean()
    return _as_uint8((_as_float(image) - grey) * factor + grey)


def fog(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Depth-independent fog: a bright veil blended over the scene.

    A true fog model attenuates with depth, which needs a depth map IDD does not provide.
    This is a uniform approximation and is described as such in RESULTS.md.
    """
    strength = [0.15, 0.26, 0.38, 0.52, 0.68][severity - 1]
    veil = np.full_like(_as_float(image), 0.85)
    return _as_uint8(_as_float(image) * (1 - strength) + veil * strength)


def rain(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Rain streaks plus mild contrast loss."""
    import cv2

    count = [220, 500, 900, 1500, 2400][severity - 1]
    height, width = image.shape[:2]
    rng = np.random.default_rng(severity)
    layer = np.zeros((height, width), dtype=np.float32)
    xs = rng.integers(0, width, count)
    ys = rng.integers(0, height, count)
    length = max(4, height // 30)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x - 1), int(min(height - 1, y + length))), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None]
    wet = _as_float(image) * 0.88 + 0.06
    return _as_uint8(wet * (1 - layer) + layer * 0.85)


def jpeg(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Compression artefacts from an aggressive transport codec."""
    import cv2

    quality = [45, 30, 20, 13, 8][severity - 1]
    ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)[:, :, ::-1]


#: Corruption name -> function. Grouped conceptually: noise, blur, photometric, weather, codec.
CORRUPTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "low_light": low_light,
    "contrast": contrast,
    "fog": fog,
    "rain": rain,
    "jpeg": jpeg,
}

#: Corruptions that plausibly occur in real driving footage, used for the headline
#: "adverse conditions" aggregate. The rest are sensor/codec artefacts.
WEATHER_LIKE = ("fog", "rain", "low_light", "motion_blur")


def make_corruption(name: str, severity: int) -> Callable[[np.ndarray], np.ndarray]:
    """Bind a corruption and severity into a single-argument callable for the dataset."""
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {name!r}; expected one of {sorted(CORRUPTIONS)}")
    if not 1 <= severity <= 5:
        raise ValueError(f"Severity must be in 1..5, got {severity}")
    function = CORRUPTIONS[name]
    return lambda image: function(image, severity)


__all__ = ["CORRUPTIONS", "WEATHER_LIKE", "make_corruption"]
