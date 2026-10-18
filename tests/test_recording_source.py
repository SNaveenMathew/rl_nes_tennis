from pathlib import Path


def test_record_bridge_supports_human_input_mode():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "MESEN_RL_RECORD_MODE" in text
    assert "emu.getInput(0)" in text
    assert "interval_action" in text
    assert "human_action" in text


def test_recording_does_not_change_scripted_input_path():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "emu.setInput(0, current_input())" in text


def test_client_accepts_optional_human_action_field():
    text = Path("tennis_rl/mesen_client.py").read_text(encoding="utf-8")
    assert 'human_action = int(parts[9]) if len(parts) >= 10 else 0' in text


def test_record_mode_preserves_auto_boot():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "RECORD_MODE and boot_complete" in text
    assert "automatic boot" in text


def test_record_cli_supports_receive_first_mode():
    text = Path("tennis_rl/record.py").read_text(encoding="utf-8")
    assert 'choices=("serve-first", "receive-first")' in text
    assert 'effective_match_games = 2' in text
    assert 'record_game_number = 2' in text


def test_receive_first_records_only_game_two():
    text = Path("tennis_rl/record.py").read_text(encoding="utf-8")
    assert 'recording_started = serve_mode == "serve-first"' in text
    assert 'game_total >= 1' in text
    assert 'game_total >= 2' in text
    assert 'recorded_game_outcome' in text


def test_record_powershell_exposes_serve_mode():
    text = Path("scripts/record.ps1").read_text(encoding="utf-8")
    assert '[ValidateSet("serve-first", "receive-first")]' in text
    assert '--serve $Serve' in text
