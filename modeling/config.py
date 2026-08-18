"""Dataclass-backed configuration loading for RoadSense.

Every experiment in this project is driven by a YAML file under ``configs/`` rather than
hardcoded constants, so that ablations differ only in their config and the exact settings
of any run can be reconstructed from ``results/runs.csv``.

Typical use::

    cfg = load_config("configs/idd_3class.yaml", train={"loss": "dice", "seed": 1})
    print(cfg.train.epochs, cfg.data.composite_fraction)
"""

from __future__ import annotations

import dataclasses
import os
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    import torch

import yaml

T = TypeVar("T")

# Repository root, resolved from this file so configs work regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Where the IDD data lives and how the prepared dataset is built.

    Every dataset in this project comes from the IDD family -- there are no external
    data sources at all.
    """

    #: Root of the IDD tree (contains ``leftImg8bit`` + ``gtFine``).
    idd_root: str = "data/raw/idd_seg"
    #: Where prepared images/masks and dataset YAMLs are written.
    out_root: str = "data/processed"

    #: ``level1`` = the 7-class main task; ``binary`` = drivable/non-drivable baseline.
    target: str = "level1"
    #: Fraction of *drive sequences* held out for validation in the sequence split.
    #: Whole sequences move together, never individual frames -- see data/sequence_split.py.
    val_sequence_fraction: float = 0.15
    #: Longest image side; ``None`` keeps IDD Lite's native 320x227.
    max_size: int | None = None

    #: Class names, index-aligned with the values written into the PNG masks.
    #: Defaults to IDD's own level-1 hierarchy.
    class_names: list[str] = field(
        default_factory=lambda: [
            "drivable",
            "non-drivable",
            "living-thing",
            "vehicles",
            "barrier-structures",
            "construction-vegetation",
            "sky",
        ]
    )
    #: Written where IDD marks a pixel void; excluded from the loss and from metrics.
    ignore_index: int = 255

    def resolved(self, name: str) -> Path:
        """Return a path field as an absolute :class:`Path` under the repo root."""
        value = Path(getattr(self, name))
        return value if value.is_absolute() else REPO_ROOT / value


@dataclass
class TrainConfig:
    """Hyperparameters for a single training run.

    Field names match the keyword arguments of :func:`modeling.train.train`, so a config
    can be splatted straight into it without a translation layer that could silently drop
    a setting.
    """

    #: Decoder architecture; see :data:`modeling.architectures.SMP_ARCHITECTURES`.
    architecture: str = "deeplabv3plus"
    #: Encoder/backbone name, e.g. ``resnet34`` / ``resnet18`` / ``mobilenet_v2``.
    encoder: str = "resnet34"
    #: ``imagenet`` for pretrained encoder weights, or ``null`` for random init.
    encoder_weights: str | None = "imagenet"

    #: One of ``ce`` / ``weighted_ce`` / ``dice`` / ``tversky`` / ``boundary``.
    loss: str = "ce"
    #: Blend weight when a region loss is combined with cross-entropy.
    loss_alpha: float = 0.5
    #: Tversky false-negative weight; 0.5 recovers Dice.
    tversky_beta: float = 0.7

    epochs: int = 40
    batch_size: int = 8
    #: ``(height, width)``; both must be multiples of 32.
    imgsz: tuple[int, int] = (224, 320)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    #: Stop after this many epochs without a new best validation mIoU.
    patience: int = 12
    #: Fraction of total steps spent in linear warmup before cosine decay.
    warmup_frac: float = 0.05
    #: Max gradient norm; 0 disables clipping.
    grad_clip: float = 1.0

    seed: int = 0
    #: ``auto`` resolves to mps > cuda > cpu.
    device: str = "auto"
    #: Dataloader workers. 0 by default -- single-process augmented loading measures
    #: ~640 img/s against ~37 img/s of MPS training throughput.
    workers: int = 0


@dataclass
class Config:
    """A complete experiment: a name, its data recipe, and its training recipe."""

    name: str = "unnamed"
    #: Free-text note recorded alongside metrics in ``results/runs.csv``.
    note: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce(value: Any, target: Any) -> Any:
    """Coerce a YAML scalar/sequence to the annotated dataclass field type."""
    origin = get_origin(target)
    if origin is tuple:
        args = get_args(target)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in value)
        return tuple(_coerce(v, a) for v, a in zip(value, args))
    if origin is list:
        (arg,) = get_args(target) or (Any,)
        return [_coerce(v, arg) for v in value]
    if is_dataclass(target) and isinstance(value, Mapping):
        return _build(target, value)
    if origin is not None and value is not None:
        # Unwrap unions such as `str | None`: coerce to the first non-None member.
        args = [a for a in get_args(target) if a is not type(None)]
        if len(args) == 1:
            return _coerce(value, args[0])
        return value
    if target in (int, float, str, bool) and value is not None:
        return target(value)
    return value


def _build(dc_type: type[T], mapping: Mapping[str, Any]) -> T:
    """Instantiate a (possibly nested) dataclass from a plain mapping.

    Field types are resolved with :func:`typing.get_type_hints` rather than read off
    ``Field.type``. Because this module uses ``from __future__ import annotations``,
    ``Field.type`` is the *string* ``"TrainConfig"``, not the class -- so a naive
    implementation never recognises a nested dataclass and silently hands back a plain
    dict, which then fails far away from the cause.

    Unknown keys raise rather than being silently dropped: a typo in a config that quietly
    reverts a hyperparameter to its default is the kind of bug that invalidates an entire
    ablation table without ever failing loudly.
    """
    known = {f.name: f for f in fields(dc_type)}
    unknown = set(mapping) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} for {dc_type.__name__}. "
            f"Valid keys: {sorted(known)}"
        )
    hints = get_type_hints(dc_type)
    kwargs = {k: _coerce(v, hints.get(k, known[k].type)) for k, v in mapping.items()}
    return dc_type(**kwargs)


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a :class:`Config` from YAML, applying keyword overrides on top.

    Args:
        path: Path to a YAML config. Relative paths resolve against the repo root.
            If ``None``, all defaults are used.
        **overrides: Nested overrides merged over the file, e.g.
            ``train={"seed": 2}``. Used by the ablation drivers to vary one axis
            at a time without writing a YAML file per cell.

    Returns:
        The fully populated config.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _build(Config, raw)


def to_dict(cfg: Config) -> dict[str, Any]:
    """Flatten a config to ``{"train.seed": 0, ...}`` for CSV experiment logging."""
    flat: dict[str, Any] = {}

    def walk(obj: Any, prefix: str = "") -> None:
        if is_dataclass(obj):
            for f in fields(obj):
                walk(getattr(obj, f.name), f"{prefix}{f.name}.")
        else:
            flat[prefix.rstrip(".")] = obj

    walk(cfg)
    return flat


def resolve_device(requested: str = "auto") -> "torch.device":
    """Resolve ``auto`` to the best available torch device.

    Ordered mps > cuda > cpu, since this project's reference machine is Apple silicon.
    Defined here rather than in :mod:`modeling.train` so that evaluation and serving code
    can resolve a device without importing the whole training stack.
    """
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects a run.

    ``deterministic`` also fixes cuDNN's algorithm choice, because the multi-seed spread
    reported in RESULTS.md is meant to measure initialisation and data-ordering variance,
    not kernel nondeterminism.
    """
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "Config",
    "DataConfig",
    "TrainConfig",
    "load_config",
    "to_dict",
    "resolve_device",
    "set_seed",
    "REPO_ROOT",
]
