from pathlib import Path


def test_record_bridge_supports_human_input_mode():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "MESEN_RL_RECORD_MODE" in text
    assert "emu.getInput(0)" in text
    assert "interval_action" in text
    assert "human_action" in text


def test_recording_does_not_change_scripted_input_path():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "emu.setInput(current_input(), 0)" in text
    assert "emu.setInput(0, current_input())" not in text


def test_client_accepts_optional_human_action_field():
    text = Path("tennis_rl/mesen_client.py").read_text(encoding="utf-8")
    assert 'human_action = int(parts[9]) if len(parts) >= 10 else 0' in text


def test_record_mode_preserves_auto_boot():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "RECORD_MODE and boot_complete" in text
    assert "automatic boot" in text



def test_automatic_boot_uses_single_frame_menu_pulses_and_requested_level():
    bridge = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    # First Start advances from the fresh title screen into difficulty selection.
    assert "elseif boot_frame == 61 then" in bridge
    assert "local level_nav_start = 102" in bridge
    assert "local nav_gap = 16" in bridge
    assert "local confirm_frame = level_nav_start + (GAME_LEVEL - 1) * nav_gap" in bridge
    assert "if boot_frame == confirm_frame then" in bridge
    # The previous 15-frame holds could straddle/skip menu transitions.
    assert "Press Start (15 frames)" not in bridge
    client = Path("tennis_rl/mesen_client.py").read_text(encoding="utf-8")
    assert '"MESEN_RL_AUTO_BOOT": "1"' in client
    assert '"MESEN_RL_LEVEL": str(self.game_level)' in client


def test_record_cli_supports_receive_first_mode():
    text = Path("tennis_rl/record.py").read_text(encoding="utf-8")
    assert 'choices=("serve-first", "receive-first")' in text
    assert 'resolve_rom' in text
    assert 'cfg.mesen.serve_mode = serve_mode' in text


def test_receive_first_uses_patched_rom_directly():
    text = Path("tennis_rl/record.py").read_text(encoding="utf-8")
    assert 'recording_started = True' in text
    assert 'recorded_game_outcome' in text


def test_record_powershell_exposes_serve_mode():
    text = Path("scripts/record.ps1").read_text(encoding="utf-8")
    assert '[ValidateSet("serve-first", "receive-first")]' in text
    assert '--serve $Serve' in text


def test_native_rgb_capture_is_record_mode_only_and_stored_separately():
    bridge = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    client = Path("tennis_rl/mesen_client.py").read_text(encoding="utf-8")
    recorder = Path("tennis_rl/record.py").read_text(encoding="utf-8")
    assert 'MESEN_RL_CAPTURE_NATIVE' in bridge
    assert 'local CAPTURE_NATIVE_FRAMES = RECORD_MODE and' in bridge
    assert 'local function capture_frame_rgb()' in bridge
    assert 'header .. frame .. native_frame' in bridge
    assert 'native_frame = np.frombuffer(payload[gray_bytes:]' in client
    assert '"native_frames"' in recorder
    assert 'native_frames=p.get("native_frames")' in recorder


def test_modern_mesen_input_api_order_is_used_for_automatic_boot():
    bridge = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    client = Path("tennis_rl/mesen_client.py").read_text(encoding="utf-8")
    # MesenCE 2.x uses setInput(input_table, port, subPort); wrong order fails
    # the inputPolled callback and prevents the Start/level/Start boot sequence.
    assert "emu.setInput(current_input(), 0)" in bridge
    assert "emu.setInput(0, current_input())" not in bridge
    assert '"MESEN_RL_LEVEL": str(self.game_level)' in client
    assert '"MESEN_RL_AUTO_BOOT": "1"' in client
