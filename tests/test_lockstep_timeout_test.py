import os, sys, time, threading, tempfile
from pathlib import Path
import pytest
lupa = pytest.importorskip("lupa")

from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lua_test_support import install_lua_test_environment
ROOT = HERE.parent
BRIDGE = str(ROOT / "mesen" / "bridge.lua")

def run(delays, slice_s=None, frame_skip=4):
    ipc = Path(tempfile.mkdtemp())
    env = {"MESEN_RL_IPC_DIR": str(ipc), "MESEN_RL_W": "8", "MESEN_RL_H": "8", "MESEN_RL_FRAME_SKIP": str(frame_skip),
           "MESEN_RL_AUTO_BOOT": "0", "MESEN_RL_LOCKSTEP": "1", "MESEN_RL_LOCKSTEP_TIMEOUT": "600",
           "MESEN_RL_RECORD_MODE": "0", "MESEN_RL_CAPTURE_NATIVE": "0", "MESEN_RL_STRICT_LOCKSTEP": "0"}
    if slice_s is not None: env["MESEN_RL_LOCKSTEP_SLICE"] = str(slice_s)
    os.environ.pop("MESEN_RL_LOCKSTEP_SLICE", None); os.environ.update(env)
    lua = lupa.LuaRuntime()
    install_lua_test_environment(lua, env)
    lua.execute((HERE / "mock_emu.lua").read_text())
    lua.execute(Path(BRIDGE).read_text())
    mock = lua.globals().mock
    st = {"done": False, "frames": []}
    def send(line):
        tmp = ipc / "h.tmp"; tmp.write_text(line + "\n"); os.replace(tmp, ipc / "command.in")
    def wait_obs(tok, timeout=20):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if (ipc / f"observation_ready_{tok}.ready").exists(): return True
            if st.get('stopped'): return False
            time.sleep(0.002)
        return False
    def host():
        send("A t0 0"); ok = wait_obs("t0")
        for i, d in enumerate(delays):
            if not ok: break
            time.sleep(d); send(f"A t{i+1} 0"); ok = wait_obs(f"t{i+1}")
        st["ok"] = ok; st["done"] = True
    th = threading.Thread(target=host); th.start()
    t0 = time.time()
    while not st["done"] and not st.get('stopped') and time.time() - t0 < 40:
        mock.run_frame()
        if mock.stopped is not None: st['stopped'] = True
    th.join()
    st["frames"] = [int(x) for x in mock.obs_frames.values()]
    err = (ipc / "error.txt").read_text() if (ipc / "error.txt").exists() else ""
    return st, err, int(mock.timeout_hits), mock.stopped, list(mock.callback_errors)

FRAME_SKIP = 4
delays = [0.0, 0.05, 2.5, 0.0, 1.2, 0.0]
st, err, hits, stopped, cberr = run(delays, frame_skip=FRAME_SKIP)
gaps = [b - a for a, b in zip(st["frames"], st["frames"][1:])]
print("host delays (s):", delays)
print("frames between observations:", gaps, "| timeout hits:", hits, "| stopped:", stopped, "| error.txt:", repr(err))
assert st["ok"] and hits == 0 and stopped is None and not err and not cberr
assert gaps[0] == FRAME_SKIP and gaps[1] == FRAME_SKIP, "fast replies must land on exactly frame_skip frames"
assert gaps[3] == FRAME_SKIP and gaps[5] == FRAME_SKIP, "after a long pause the next step is back to exactly frame_skip frames"
assert FRAME_SKIP <= gaps[2] <= FRAME_SKIP + 14 and FRAME_SKIP <= gaps[4] <= FRAME_SKIP + 8, "a long pause may add zero or only a handful of idle frames"
print(f"2.5s pause cost {gaps[2]-FRAME_SKIP} idle frames, 1.2s pause cost {gaps[4]-FRAME_SKIP} (an unthrottled emulator would run ~100,000s)")

# Control: the old unsliced wait (one callback that spins until the host answers) must hit the timeout.
st2, err2, hits2, stopped2, cberr2 = run([0.0, 2.5, 0.0], slice_s=100)
print(f"CONTROL (slice=100s, i.e. the old design): timeout hits={hits2} stopped={stopped2} error.txt={err2.strip()[:70]!r}")
assert hits2 >= 1 and (stopped2 is not None or "Maximum execution time" in err2 or cberr2)
print("LOCKSTEP TIMEOUT OK")
