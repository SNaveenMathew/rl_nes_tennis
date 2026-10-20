from __future__ import annotations

import pytest

from tennis_rl.config import Config
from tennis_rl.inference_compat import validate_checkpoint_runtime


def _saved_config(*, width=128, height=120, frame_skip=2, pulses=(1, 2)):
    return {
        "mesen": {
            "observation_width": width,
            "observation_height": height,
            "frame_skip": frame_skip,
        },
        "hierarchical": {"button_pulse_durations": list(pulses)},
    }


def test_hierarchical_checkpoint_runtime_matches_config():
    cfg = Config()
    payload = {
        "extra": {"config": _saved_config()},
        "button_pulse_durations": [1, 2],
    }
    assert validate_checkpoint_runtime(payload, cfg, hierarchical=True) == (1, 2)


@pytest.mark.parametrize(
    ("field", "saved", "phrase"),
    [
        ("observation_width", 84, "observation_width"),
        ("observation_height", 84, "observation_height"),
        ("frame_skip", 4, "frame_skip"),
    ],
)
def test_hierarchical_checkpoint_rejects_mismatched_runtime_contract(field, saved, phrase):
    cfg = Config()
    saved = {"mesen": {field: saved}}
    with pytest.raises(ValueError, match=phrase):
        validate_checkpoint_runtime({"extra": {"config": saved}, "button_pulse_durations": [1, 2]}, cfg, hierarchical=True)


def test_hierarchical_checkpoint_rejects_pulses_longer_than_frame_skip():
    cfg = Config()
    with pytest.raises(ValueError, match="exceed active frame_skip"):
        validate_checkpoint_runtime({"button_pulse_durations": [1, 2, 4]}, cfg, hierarchical=True)


def test_flat_checkpoint_settings_are_checked_when_saved():
    cfg = Config()
    with pytest.raises(ValueError, match="frame_skip"):
        validate_checkpoint_runtime({"config": _saved_config(frame_skip=4)}, cfg, hierarchical=False)


def test_legacy_checkpoint_without_config_remains_checkable_by_weights():
    cfg = Config()
    assert validate_checkpoint_runtime({"model": {}}, cfg, hierarchical=False) is None
