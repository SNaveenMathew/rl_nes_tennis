from pathlib import Path


def test_bridge_uses_fast_raw_grayscale_io_transport():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert 'require("socket")' not in text
    assert "MESEN_RL_IPC_DIR" in text
    assert "io.open" in text
    assert "emu.setInput" in text
    assert "emu.getScreenBuffer" in text
    assert "takeScreenshot" not in text
    assert "getScreenSize" not in text
    assert "local SRC_W = 256" in text
    assert "local SRC_H = 240" in text
    assert "table.concat(out)" in text


def test_bridge_source_has_bounded_command_mailbox_and_host_token_observations():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert 'COMMAND_PATH = IPC_DIR .. "/command.in"' in text
    assert 'observation_"' in text
    assert "next_command_seq" not in text
    assert "pending_token" in text and "token" in text


def test_ready_handshake_does_not_require_initial_observation():
    text = Path("tennis_rl/mesen_client.py").read_text(encoding="utf-8")
    startup = text[text.index("def _start"):text.index("def _read_stderr_tail")]
    assert "self._consume_observation()" not in startup


def test_bridge_emits_after_command_skip_not_initial_ready_frame():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "initial_observation_sent" not in text
    assert "if frame_counter % FRAME_SKIP == 0 then" not in text


def test_bridge_polls_commands_every_frame_and_emits_one_observation_per_command():
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "poll_command()" in source
    assert "local step_pending = false" in source
    assert "if step_pending then" in source
    assert "if frame_counter >= FRAME_SKIP then" in source
    assert "step_pending = false" in source
    assert "emu.reset()" not in source


def test_observation_filenames_match_host_token_format():
    src = (Path(__file__).parents[1] / "mesen" / "bridge.lua").read_text()
    assert "pending_token" in src and "token" in src
    assert 'tostring(seq) .. ".bin"' not in src


def test_bridge_publishes_observation_ready_marker():
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "OBS_READY_PREFIX" in source
    assert "observation_ready_" in source


def test_bridge_uses_latest_command_mailbox():
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "local function poll_command()" in source
    assert "local line = read_command()" in source
    assert "apply_command(line)" in source
    assert "while true do" not in source[source.index("local function poll_command()"):source.index("local function current_input()", source.index("local function poll_command()"))]


def test_bridge_ram_addresses_and_game_loss():
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "RAM_PLAYER_SCORE = 0x0058" in source
    assert "RAM_OPPONENT_SCORE = 0x0059" in source
    assert "REWARD_GAME_LOSS" in source


def test_bridge_score_reset_game_tracking_and_sticky_peaks():
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert "actual_player_games" in source
    assert "actual_opponent_games" in source
    assert "max_player_score_seen" in source
    assert "max_opponent_score_seen" in source
    assert "game_ended" in source


def test_bridge_normalizes_windows_crlf_and_savestate_paths():
    """CRLF commands must not leave a carriage return attached to a state path."""
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert 'line = line:gsub("\\r$", "")' in source
    assert "normalize_file_path(rest)" in source
    assert "normalize_file_path(path)" in source
    assert 'local IPC_DIR = (os.getenv("MESEN_RL_IPC_DIR") or "runs/mesen_ipc/worker"):gsub("\\\\", "/")' in source


def test_load_from_noop_completion_is_a_lockstep_frame_barrier():
    """Do not let asynchronous host wake-up add arbitrary frames after replay."""
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    start = source.index("-- No-op frames after LOAD_FROM run free")
    end = source.index("-- Count down precise A/B pulses", start)
    block = source[start:end]
    ack = block.index('write_text(IPC_DIR .. "/state_loaded.txt", "LOADED\\n")')
    lockstep = block.index("if LOCKSTEP then", ack)
    wait = block.index("lockstep_wait()", lockstep)
    early_return = block.index("return", wait)
    assert ack < lockstep < wait < early_return


def test_strict_lockstep_fails_closed_instead_of_advancing_on_host_delay():
    source = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    assert 'MESEN_RL_STRICT_LOCKSTEP") == "1"' in source
    assert "strict lockstep host-command timeout" in source
    assert "emu.stop(38)" in source
    assert "return true  -- soft-lockstep yields one frame" in source


def test_api_info_is_required_only_for_current_bridge(tmp_path):
    """Legacy bridge fixtures need not implement current diagnostic metadata."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from lua_test_support import read_api_info_for_bridge

    current = Path("mesen/bridge.lua").resolve()
    legacy = Path("tests/fixtures/original_bridge.lua").resolve()
    assert read_api_info_for_bridge(tmp_path, legacy, current) == ""

    try:
        read_api_info_for_bridge(tmp_path, current, current)
    except AssertionError as exc:
        assert "did not write api_info.txt" in str(exc)
    else:
        raise AssertionError("current bridge metadata absence should be a test failure")

    (tmp_path / "api_info.txt").write_text("lockstep=true\nstrict_lockstep=true wait=0.7s\n", encoding="utf-8")
    assert "strict_lockstep=true" in read_api_info_for_bridge(tmp_path, current, current)
