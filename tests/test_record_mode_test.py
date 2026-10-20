"""Exact path of the user's failure: record mode (human plays), auto-boot, repeated SAVE_TO."""
import os, sys, time, threading, tempfile
from pathlib import Path
import pytest
lupa = pytest.importorskip("lupa")

from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lua_test_support import install_lua_test_environment

from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
ROOT = HERE.parent
BRIDGE = str(ROOT / "mesen" / "bridge.lua")

def run(bridge=BRIDGE, prelude="", ack_timeout=5):
    ipc = Path(tempfile.mkdtemp())
    env = {"MESEN_RL_IPC_DIR": str(ipc), "MESEN_RL_W": "8", "MESEN_RL_H": "8", "MESEN_RL_FRAME_SKIP": "4",
        "MESEN_RL_AUTO_BOOT": "1", "MESEN_RL_RECORD_MODE": "1", "MESEN_RL_LOCKSTEP": "1",
        "MESEN_RL_CAPTURE_NATIVE": "0", "MESEN_RL_STRICT_LOCKSTEP": "0"}
    os.environ.update(env)
    lua = lupa.LuaRuntime()
    install_lua_test_environment(lua, env)
    lua.execute((HERE / "mock_emu.lua").read_text())
    lua.execute(prelude)
    lua.execute(Path(bridge).read_text())
    mock = lua.globals().mock
    res = {"done": False}
    def host():
        t0 = time.time(); ready = False
        while time.time() - t0 < 10:
            p = ipc / "ready.txt"
            if p.exists() and p.read_bytes().startswith(b"READY"): ready = True; break
            time.sleep(0.005)
        res["ready"] = ready
        for k in range(4):                        # four SAVE_TOs while the "human" plays
            ack = ipc / "state_saved.txt"; ack.unlink(missing_ok=True)
            tmp = ipc / "h.tmp"; tmp.write_text(f"SAVE_TO {ipc}/s{k}.state\n"); os.replace(tmp, ipc / "command.in")
            t1 = time.time()
            while not ack.exists() and time.time() - t1 < ack_timeout: time.sleep(0.002)
            res[f"ack{k}"] = ack.exists() and (ipc / f"s{k}.state").exists()
            time.sleep(0.05)
        res["bytes"] = (ipc / "s0.state").read_bytes() if (ipc / "s0.state").exists() else b""
        res["done"] = True
    th = threading.Thread(target=host); th.start()
    t0 = time.time()
    while not res["done"] and time.time() - t0 < 30: mock.run_frame()
    th.join()
    err = (ipc / "error.txt").read_text() if (ipc / "error.txt").exists() else ""
    info = (ipc / "api_info.txt").read_text() if (ipc / "api_info.txt").exists() else ""
    return res, err, info, int(mock.timeout_hits), mock.stopped

res, err, info, hits, stopped = run()
print({k: v for k, v in res.items() if k != "bytes"}, "| error:", repr(err), "| timeout_hits:", hits)
assert res["ready"] and all(res[f"ack{k}"] for k in range(4)) and not err and hits == 0 and stopped is None
assert res["bytes"].endswith(b"\0binary\0tail"), "binary state must survive file write intact (embedded NULs)"
print("api_info.txt ->", info.strip().splitlines()[-2])

# A Mesen that refuses savestates from startFrame must give a readable error, not crash/hang.
class_strict = 'mock.strict_startframe = true'
res2, err2, _, _, stopped2 = run(prelude='local f = emu.saveSavestate; function emu.saveSavestate() error("refused by this build") end', ack_timeout=0.6)
print("refusing build ->", {k: v for k, v in res2.items() if k != "bytes"}, "| error:", repr(err2.strip()[:90]))
assert not res2["ack0"] and "refused by this build" in err2 and stopped2 is None
print("RECORD MODE OK")
