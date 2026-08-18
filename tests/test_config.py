"""Config-loading tests.

The loader is small but two of its behaviours matter a lot: nested dataclasses must
actually be built (not left as dicts), and unknown keys must raise. A typo that silently
reverts a hyperparameter to its default would invalidate an ablation table without ever
failing loudly.
"""

from __future__ import annotations

import pytest

from modeling.config import (
    Config,
    DataConfig,
    TrainConfig,
    load_config,
    resolve_device,
    set_seed,
    to_dict,
)


def test_defaults_are_typed_dataclasses():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.train, TrainConfig)


def test_nested_dataclasses_are_built_from_yaml(tmp_path):
    """Regression: `from __future__ import annotations` makes Field.type a string, so a
    loader reading Field.type directly hands back plain dicts for nested sections."""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "name: t\ntrain:\n  loss: dice\n  epochs: 7\ndata:\n  target: binary\n"
    )
    cfg = load_config(path)
    assert isinstance(cfg.train, TrainConfig), "nested section was not built"
    assert isinstance(cfg.data, DataConfig)
    assert cfg.train.loss == "dice" and cfg.train.epochs == 7
    assert cfg.data.target == "binary"


def test_unspecified_fields_keep_their_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  loss: tversky\n")
    cfg = load_config(path)
    assert cfg.train.loss == "tversky"
    assert cfg.train.architecture == TrainConfig().architecture
    assert cfg.train.lr == TrainConfig().lr


def test_sequence_fields_coerce_to_tuple(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  imgsz: [256, 384]\n")
    cfg = load_config(path)
    assert cfg.train.imgsz == (256, 384)
    assert isinstance(cfg.train.imgsz, tuple)


def test_optional_field_accepts_null(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  encoder_weights: null\n")
    assert load_config(path).train.encoder_weights is None


def test_unknown_key_raises_rather_than_being_dropped():
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(None, train={"lernign_rate": 0.1})
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(None, data={"not_a_field": 1})


def test_keyword_overrides_are_merged_over_the_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  loss: dice\n  epochs: 30\n")
    cfg = load_config(path, train={"seed": 3})
    assert cfg.train.seed == 3
    assert cfg.train.loss == "dice"  # untouched keys survive the merge
    assert cfg.train.epochs == 30


def test_to_dict_flattens_for_csv_logging():
    flat = to_dict(load_config())
    assert flat["train.loss"] == "ce"
    assert "data.target" in flat
    assert all("." in k or k in {"name", "note"} for k in flat)


def test_resolved_returns_absolute_paths():
    cfg = load_config()
    assert cfg.data.resolved("idd_root").is_absolute()


def test_resolve_device_and_set_seed():
    import torch

    assert isinstance(resolve_device("cpu"), torch.device)
    assert resolve_device("cpu").type == "cpu"
    set_seed(0)
    first = torch.randn(4)
    set_seed(0)
    assert torch.equal(first, torch.randn(4)), "set_seed did not make sampling reproducible"
