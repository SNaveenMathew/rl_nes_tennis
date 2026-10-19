import os, sys, time, threading, shutil, tempfile
from pathlib import Path
import pytest
lupa = pytest.importorskip("lupa")

from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from lua_test_support import install_lua_test_environment

def run(bridge_path, script, *, strict=False):
    ipc = Path(tempfile.mkdtemp())
    env = {"MESEN_RL_IPC_DIR": str(ipc), "MESEN_RL_W": "8", "MESEN_RL_H": "8",
        "MESEN_RL_FRAME_SKIP": "4", "MESEN_RL_AUTO_BOOT": "0", "MESEN_RL_LOCKSTEP": "1",
        "MESEN_RL_LOCKSTEP_TIMEOUT": "20", "MESEN_RL_STRICT_LOCKSTEP": "1" if strict else "0",
        "MESEN_RL_STRICT_WAIT_S": "0.70"}
    os.environ.update(env)
    # The record-mode test runs during pytest collection and leaves its process
    # environment mutated. Explicit injection prevents its RECORD_MODE=1 from
    # leaking here (RECORD_MODE intentionally disables lockstep in the bridge).
    # It also avoids relying on embedded Lua seeing Python runtime env changes.
    lua = lupa.LuaRuntime()
    install_lua_test_environment(lua, env)
    if strict:
        # The embedded-Lua test and host thread share a Python process. Yield the
        # Python scheduler during strict lockstep polling so the host can publish
        # its next command, just as it can when Mesen runs in a separate process.
        # This does not advance mock emulator frames and does not relax the strict
        # frame-boundary semantics or timeout.
        lua.globals()["__MESEN_RL_TEST_YIELD"] = lambda: time.sleep(0.001)
    lua.execute((HERE / "mock_emu.lua").read_text())
    lua.execute(Path(bridge_path).read_text())
    mock = lua.globals().mock
    api_info = (ipc / "api_info.txt").read_text(encoding="utf-8")
    if bridge_path == str(ROOT / "mesen" / "bridge.lua"):
        expected_strict = "strict_lockstep=true" if strict else "strict_lockstep=false"
        assert "lockstep=true" in api_info and expected_strict in api_info, (
            "test configuration did not reach Lua; expected lockstep plus "
            f"{expected_strict}. api_info.txt:\n{api_info}"
        )
    state = {"done": False, "log": []}

    def send(line):
        tmp = ipc / "h.tmp"; tmp.write_text(line + "\n"); os.replace(tmp, ipc / "command.in")
    def wait(name, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if (ipc / name).exists(): return True
            time.sleep(0.001)
        return False
    def step(tok):
        # Tests deliberately reuse the "idle" token below. Remove the prior ready
        # marker before issuing a new command so an old marker cannot satisfy the
        # next wait before the new observation has been produced.
        ready = ipc / f"observation_ready_{tok}.ready"
        ready.unlink(missing_ok=True)
        send(f"A {tok} 0")
        assert wait(f"observation_ready_{tok}.ready"), "no observation"
        return None
    ctx = dict(send=send, wait=wait, step=step, ipc=ipc, state=state, mock=mock)

    # Run script-side assertions in a wrapper that always signals completion.
    # Previously an exception in the helper thread left `done` false, so the
    # mock emulator kept ticking for 40 seconds and the test later failed with
    # a misleading KeyError for a result the script never populated.
    def run_script():
        try:
            script(ctx)
        except BaseException as exc:
            state["script_exception"] = exc
        finally:
            state["done"] = True

    th = threading.Thread(target=run_script, name="state-ops-test-script")
    th.start()
    t0 = time.time()
    while not state["done"] and time.time() - t0 < 40:
        mock.run_frame()
    th.join()
    if "script_exception" in state:
        exc = state["script_exception"]
        raise AssertionError(f"state-ops script thread failed: {exc!r}") from exc
    err = ""
    p = ipc / "error.txt"
    if p.exists(): err = p.read_text()
    # Query Lua only from the emulation-driving thread, never the host helper.
    state["active_callbacks"] = int(mock.active_callbacks())
    out = (state, int(mock.frame), int(mock.loaded or 0), err, ipc)
    return out

# ---------------- NEW bridge ----------------
def script_new(c):
    ipc, st, mock = c["ipc"], c["state"], c["mock"]
    frames = []
    def obs(tok):
        c["step"](tok); frames.append((tok, "obs")); st["log"].append(("obs", tok))
    # frame counter is only read from the host side *between* steps, when the emulator is parked
    for i in range(3): obs(f"a{i}")
    c["send"](f"SAVE_TO {ipc}/s.state")
    assert c["wait"]("state_saved.txt"), "SAVE_TO never acknowledged"
    st["saved"] = (ipc / "s.state").read_text()
    (ipc / "state_saved.txt").unlink()
    for i in range(5): obs(f"b{i}")
    c["send"](f"LOAD_FROM 7 {ipc}/s.state")
    assert c["wait"]("state_loaded.txt"), "LOAD_FROM never acknowledged"
    (ipc / "state_loaded.txt").unlink()
    st["after_load_noop7"] = True
    obs("c0")
    c["send"](f"LOAD_FROM 0 {ipc}/s.state")
    assert c["wait"]("state_loaded.txt"), "LOAD_FROM 0 never acknowledged"
    (ipc / "state_loaded.txt").unlink()
    obs("d0")

st, frame, loaded, err, ipc = run(str(ROOT / "mesen" / "bridge.lua"), script_new)
print("NEW bridge: savestate file ->", st.get("saved"), "| loads:", loaded, "| error.txt:", repr(err))
assert st["done"] and loaded == 2 and not err and st["saved"].startswith("STATE|")

# deterministic frame accounting, checked from the file the bridge wrote at SAVE_TO time
saved_frame = int(st["saved"].split("|")[1])
print("saved at emulator frame", saved_frame)
shutil.rmtree(ipc, ignore_errors=True)

# ---------------- OLD bridge (original upload) ----------------
def script_old(c):
    # This fixture is the original Mesen 0.9.x bridge, whose observation
    # mailbox protocol is intentionally different from the current bridge.
    # Do not call c["step"] here: that helper waits for the new bridge's
    # observation_ready_<token>.ready file, which the legacy fixture never
    # writes. This regression only verifies that the old savestate API fails
    # clearly instead of hanging or falsely acknowledging SV.
    c["send"]("SV")
    c["state"]["acked"] = c["wait"]("state_saved.txt", timeout=1.0)
st2, frame2, loaded2, err2, ipc2 = run(str(HERE / "fixtures" / "original_bridge.lua"), script_old)
print("ORIGINAL bridge: SV acknowledged?", st2["acked"], "| error.txt:", repr(err2.strip()))
assert st2["acked"] is False and "Failed to create savestate" in err2
shutil.rmtree(ipc2, ignore_errors=True)
print("ALL OK")

# ---------------- determinism of LOAD_FROM / steps ----------------
def script_det(c):
    ipc, st = c["ipc"], c["state"]
    def snap(name):
        c["send"](f"SAVE_TO {ipc}/{name}.state"); assert c["wait"]("state_saved.txt")
        (ipc / "state_saved.txt").unlink()
        return int((ipc / f"{name}.state").read_text().split("|")[1])
    for i in range(3): c["step"](f"a{i}")
    f0 = snap("base")
    loads = []
    for k in range(3):
        c["send"](f"LOAD_FROM 7 {ipc}/base.state"); assert c["wait"]("state_loaded.txt")
        (ipc / "state_loaded.txt").unlink()
        loads.append(snap(f"l{k}"))
    c["send"](f"LOAD_FROM 0 {ipc}/base.state"); assert c["wait"]("state_loaded.txt")
    (ipc / "state_loaded.txt").unlink()
    zero = snap("z")
    gaps = []
    prev = snap("p0")
    for i in range(4):
        __import__("time").sleep(0.15); c["step"](f"s{i}"); cur = snap(f"p{i+1}"); gaps.append(cur - prev); prev = cur
    for _ in range(3): c["step"]("idle")
    st["det"] = dict(base=f0, loads=loads, zero=zero, gaps=gaps)
st3, _, _, err3, ipc3 = run(str(ROOT / "mesen" / "bridge.lua"), script_det, strict=True)
d = st3["det"]; d["armed_when_idle"] = int(st3.get("active_callbacks", 0)); print("determinism:", d, "| error:", repr(err3))
assert d["armed_when_idle"] == 0, "savestate callback leaked while idle"
assert not err3, f"bridge emitted an error during replay determinism test: {err3!r}"
assert d["loads"] == [d["base"] + 7] * 3, (
    "LOAD_FROM 7 must always run exactly seven no-op frames before the next saved state; "
    f"base={d['base']} loads={d['loads']}"
)
assert d["zero"] == d["base"] + 1, (
    "LOAD_FROM 0 should restore the base state, then acknowledge the next startFrame boundary; "
    f"base={d['base']} zero={d['zero']}"
)
assert len(set(d["gaps"])) == 1, "every step must advance the emulator by the same number of frames"
shutil.rmtree(ipc3, ignore_errors=True)
print("DETERMINISM OK")
