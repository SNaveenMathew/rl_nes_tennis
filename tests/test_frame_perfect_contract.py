from __future__ import annotations

from pathlib import Path

import pytest

from tennis_rl.config import Config, load_config, validate_frame_perfect_config
from tennis_rl.mesen_client import MesenWorker

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_keep_full_resolution_and_one_raw_frame_per_action():
    for relative in ("configs/base.json", "configs/base.json.example", "configs/scenario.json"):
        path = ROOT / relative
        if not path.is_file():
            continue
        config = load_config(path)
        assert config.mesen.observation_width == 256
        assert config.mesen.observation_height == 240
        assert config.mesen.frame_skip == 1
        if config.hierarchical.enabled:
            assert config.hierarchical.durations == [1]
            assert config.hierarchical.button_pulse_durations == [1]


def test_config_contract_rejects_spatial_or_temporal_downsampling():
    cfg = Config()
    cfg.mesen.observation_width = 128
    with pytest.raises(ValueError, match="frame-perfect contract"):
        validate_frame_perfect_config(cfg, context="test")

    cfg = Config()
    cfg.mesen.frame_skip = 2
    with pytest.raises(ValueError, match="frame-perfect contract"):
        validate_frame_perfect_config(cfg, context="test")


def test_config_contract_rejects_multi_frame_hierarchical_actions():
    cfg = Config()
    cfg.hierarchical.durations = [1, 2]
    with pytest.raises(ValueError, match=r"hierarchical\.durations"):
        validate_frame_perfect_config(cfg, context="test")

    cfg = Config()
    cfg.hierarchical.button_pulse_durations = [1, 2]
    with pytest.raises(ValueError, match=r"button_pulse_durations"):
        validate_frame_perfect_config(cfg, context="test")


def test_worker_fails_before_emulator_launch_if_frame_perfect_contract_is_broken():
    with pytest.raises(ValueError, match="requires width=256, height=240, frame_skip=1"):
        MesenWorker(
            executable="missing.exe", rom="missing.nes", lua_script="missing.lua",
            host="127.0.0.1", port=39100, width=128, height=120, frame_skip=1,
            startup_timeout=1, socket_timeout=1,
        )


def test_bridge_records_action_and_button_duration_on_each_raw_frame():
    source = (ROOT / "mesen" / "bridge.lua").read_text(encoding="utf-8")
    callback = source[source.index("local input_callback ="):source.index("local function reset_tracking()")]
    assert 'local FRAME_SKIP = tonumber(os.getenv("MESEN_RL_FRAME_SKIP") or "1")' in source
    assert 'local WIDTH = tonumber(os.getenv("MESEN_RL_W") or "256")' in source
    assert 'local HEIGHT = tonumber(os.getenv("MESEN_RL_H") or "240")' in source
    assert "if FRAME_SKIP == 1 then\n            interval_action = human_action" in callback
    assert "if input.a == true then interval_a_frames = interval_a_frames + 1 end" in callback
    assert "if input.b == true then interval_b_frames = interval_b_frames + 1 end" in callback
    assert "point_end, interval_action, interval_a_frames, interval_b_frames" in source


def test_default_model_and_runtime_contract_have_full_raster_defaults():
    cfg = Config()
    assert (cfg.mesen.observation_width, cfg.mesen.observation_height, cfg.mesen.frame_skip) == (256, 240, 1)
    assert cfg.hierarchical.durations == [1]
    assert cfg.hierarchical.button_pulse_durations == [1]
