"""Mesen bridge regression tests for early/partial screen-buffer APIs."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
lupa = pytest.importorskip("lupa")
from lupa import LuaRuntime  # noqa: E402
from lua_test_support import install_lua_test_environment  # noqa: E402


def _wait_for(path: Path, timeout: float = 8.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path
        time.sleep(0.001)
    raise TimeoutError(f"timed out waiting for {path}")


def _run_capture_case(tmp_path: Path, *, record: bool, native: bool) -> dict:
    ipc = tmp_path / "ipc"
    ipc.mkdir()
    env = {
        "MESEN_RL_IPC_DIR": str(ipc),
        "MESEN_RL_W": "8",
        "MESEN_RL_H": "8",
        "MESEN_RL_FRAME_SKIP": "4" if not record else "2",
        "MESEN_RL_AUTO_BOOT": "0",
        "MESEN_RL_LOCKSTEP": "1",
        "MESEN_RL_LOCKSTEP_SLICE": "0.01",
        "MESEN_RL_LOCKSTEP_TIMEOUT": "20",
        "MESEN_RL_STRICT_LOCKSTEP": "0",
        "MESEN_RL_RECORD_MODE": "1" if record else "0",
        "MESEN_RL_CAPTURE_NATIVE": "1" if native else "0",
    }

    lua = LuaRuntime()
    install_lua_test_environment(lua, env)
    # Embedded-Lua mode needs to yield the Python scheduler while waiting for a
    # host mailbox command, mirroring the helper in test_state_ops_test.py.
    lua.globals()["__MESEN_RL_TEST_YIELD"] = lambda: time.sleep(0.001)
    lua.execute((ROOT / "tests" / "mock_emu.lua").read_text(encoding="utf-8"))

    if native:
        # Some builds expose only 239 zero-based rows for this 240-line console.
        # The bridge must defer while unavailable, then emit stable 256x240 RGB by
        # repeating the final visible row rather than crashing or shortening data.
        lua.execute(r"""
            local n = 0
            emu.getScreenBuffer = function()
                n = n + 1
                if n <= 4 then return nil end
                local b = {}
                for i = 0, (256 * 239) - 1 do
                    local y = math.floor(i / 256)
                    b[i] = 0xFF000000 + y * 0x10000 + 0x1234
                end
                return b
            end
        """)
    else:
        # On startup, getScreenBuffer may return nil before the PPU has rendered.
        # These first calls should be retried without losing the pending token.
        lua.execute(r"""
            local original = emu.getScreenBuffer
            local n = 0
            emu.getScreenBuffer = function()
                n = n + 1
                if n <= 4 then return nil end
                return original()
            end
        """)

    lua.execute((ROOT / "mesen" / "bridge.lua").read_text(encoding="utf-8"))
    assert (ipc / "ready.txt").is_file(), "bridge did not publish ready.txt"
    result: dict = {}
    token = "native" if native else "retry"

    def send(line: str) -> None:
        tmp = ipc / "host.tmp"
        tmp.write_text(line + "\n", encoding="utf-8")
        os.replace(tmp, ipc / "command.in")

    def host() -> None:
        try:
            send(f"A {token} 0")
            _wait_for(ipc / f"observation_ready_{token}.ready")
            raw = (ipc / f"observation_{token}.bin").read_bytes()
            header, payload = raw.split(b"\n", 1)
            result["header"] = header.decode("utf-8")
            result["payload_bytes"] = len(payload)
            if native:
                gray_len = 8 * 8
                rgb = payload[gray_len:]
                expected_rgb_bytes = 256 * 240 * 3
                assert len(payload) == gray_len + expected_rgb_bytes
                assert len(rgb) == expected_rgb_bytes
                row_bytes = 256 * 3
                # The 239 visible rows run 0..238; output row 239 is padded with
                # the last visible row. This tests image semantics, not just length.
                assert rgb[238 * row_bytes : 239 * row_bytes] == rgb[239 * row_bytes : 240 * row_bytes]
                assert rgb[0] == 0 and rgb[238 * row_bytes] == 238
            else:
                assert len(payload) == 8 * 8
            result["payload"] = payload
        except BaseException as exc:
            result["exception"] = exc
        finally:
            try:
                send("Q")
                result["shutdown_sent"] = True
            except BaseException as exc:
                result["shutdown_exception"] = exc

    th = threading.Thread(target=host, name=f"screen-buffer-test-{token}", daemon=True)
    th.start()
    mock = lua.globals().mock
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if result.get("shutdown_sent") and mock.stopped is not None:
            break
        mock.run_frame()
    th.join(timeout=2)
    try:
        assert not th.is_alive(), "host test thread did not finish"
        assert result.get("shutdown_sent"), f"host did not send shutdown: {result!r}"
        assert "exception" not in result, f"host side failed: {result.get('exception')!r}"
        assert "shutdown_exception" not in result, f"shutdown failed: {result.get('shutdown_exception')!r}"
        assert mock.stopped == 0, f"bridge did not stop cleanly: {mock.stopped!r}"
        assert int(mock.active_callbacks()) == 0, "unexpected CPU exec callback remained registered"
        assert len(mock.callback_errors) == 0, f"Lua callback exceptions: {list(mock.callback_errors)}"
        assert not (ipc / "error.txt").exists(), (
            (ipc / "error.txt").read_text(encoding="utf-8") if (ipc / "error.txt").exists() else ""
        )
        return result
    finally:
        # A LuaRuntime must not be used after its owning thread exits; the test
        # drives it only on this thread and closes it once the protocol completes.
        lua = None


def test_capture_retries_until_a_real_framebuffer_exists(tmp_path: Path):
    result = _run_capture_case(tmp_path, record=False, native=False)
    assert result["header"].startswith("O retry ")
    assert result["payload_bytes"] == 64


def test_native_capture_pads_a_239_row_frame_to_256_by_240(tmp_path: Path):
    result = _run_capture_case(tmp_path, record=True, native=True)
    assert result["header"].startswith("O native ")
    assert result["payload_bytes"] == 64 + 256 * 240 * 3
