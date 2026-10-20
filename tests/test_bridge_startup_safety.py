"""Regression guards for Mesen API callback context and early video buffers."""
from pathlib import Path


def test_savestate_ops_are_deferred_to_one_shot_cpu_exec_callback():
    source = (Path(__file__).parents[1] / "mesen" / "bridge.lua").read_text(encoding="utf-8")
    assert "local EXEC_CALLBACK_TYPE = (emu.callbackType and emu.callbackType.exec)" in source
    assert "local function perform_state_op_in_exec()" in source
    assert "local function main_cpu_program_counter()" in source
    assert "emu.addMemoryCallback(one_shot_exec_callback, EXEC_CALLBACK_TYPE, pc, pc)" in source
    assert "emu.removeMemoryCallback(id, EXEC_CALLBACK_TYPE, state_op_callback_address, state_op_callback_address)" in source
    assert "state_op_callback_id ~= nil" in source
    # Callback removal must occur outside the CPU callback dispatcher; removing
    # while Mesen iterates its memory-callback collection can invalidate it.
    callback_body = source.split("local function one_shot_exec_callback()", 1)[1].split("local ok, id = pcall", 1)[0]
    assert "removeMemoryCallback" not in callback_body
    assert "if state_op_callback_fired then" in source


def test_capture_defers_empty_framebuffers_and_retries_pending_observation():
    source = (Path(__file__).parents[1] / "mesen" / "bridge.lua").read_text(encoding="utf-8")
    assert "local function get_valid_screen_buffer()" in source
    assert "if frame == nil then return false end" in source
    assert "if native_frame == nil then return false end" in source
    assert "screen_capture_wait_frames = screen_capture_wait_frames + 1" in source
    assert "if sent then" in source
    assert "Screen buffer remained unavailable or incomplete for 180 frames" in source


def test_api_diagnostics_include_memory_callback_and_cpu_enums():
    source = (Path(__file__).parents[1] / "mesen" / "bridge.lua").read_text(encoding="utf-8")
    assert '{"eventType", "memType", "callbackType", "cpuType"}' in source
