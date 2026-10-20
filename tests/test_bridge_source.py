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


def test_strict_state_ops_harness_sends_q_after_final_observation():
    """A strict replay test must terminate the final parked bridge cleanly."""
    harness = Path("tests/test_state_ops_test.py").read_text(encoding="utf-8")
    run_start = harness.index("    def run_script():")
    driver_start = harness.index("    th = threading.Thread(target=run_script", run_start)
    cleanup = harness[run_start:driver_start]
    assert "if strict:" in cleanup
    assert 'send("Q")' in cleanup
    assert 'state["shutdown_command_sent"] = True' in cleanup
    driver_end = harness.index("    th.join()", driver_start)
    driver = harness[driver_start:driver_end]
    assert "mock.stopped is not None" in driver
    assert "strict-lockstep bridge did not consume the host Q command" in harness



def test_state_ops_harness_reads_api_info_only_for_current_bridge():
    """Legacy bridge fixtures predate api_info.txt and must not be required to create it."""
    harness = Path("tests/test_state_ops_test.py").read_text(encoding="utf-8")
    branch_start = harness.index('    if Path(bridge_path).resolve() == (ROOT / "mesen" / "bridge.lua").resolve():')
    api_path = harness.index('        api_info_path = ipc / "api_info.txt"', branch_start)
    branch_end = harness.index('    state = {"done": False, "log": []}', branch_start)
    assert branch_start < api_path < branch_end
    assert 'assert api_info_path.is_file()' in harness[branch_start:branch_end]
