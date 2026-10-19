"""Run mesen/bridge.lua against a mock emulator and check lock-step timing."""
import os, sys, time, threading, shutil, tempfile
from pathlib import Path
import pytest
lupa = pytest.importorskip("lupa")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lua_test_support import install_lua_test_environment

BRIDGE = Path(str(Path(__file__).resolve().parents[1] / "mesen" / "bridge.lua")).read_text()

def run(lockstep: bool, host_delays, frame_skip=4):
    ipc = Path(tempfile.mkdtemp())
    env = {
        "MESEN_RL_IPC_DIR": str(ipc), "MESEN_RL_W": "8", "MESEN_RL_H": "8",
        "MESEN_RL_FRAME_SKIP": str(frame_skip), "MESEN_RL_AUTO_BOOT": "0",
        "MESEN_RL_LOCKSTEP": "1" if lockstep else "0",
        "MESEN_RL_LOCKSTEP_TIMEOUT": "20", "MESEN_RL_STRICT_LOCKSTEP": "0",
    }
    os.environ.update(env)
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    install_lua_test_environment(lua, env)
    state = {"frames": 0, "stopped": None, "inputs": []}
    cbs = {}
    lua.execute("emu = {}")
    emu = lua.globals().emu
    emu.eventType = lua.table(inputPolled="inputPolled", endFrame="endFrame")
    emu.memType = lua.table(nesMemory="nes")
    emu.addEventCallback = lambda fn, t: cbs.__setitem__(t, fn) or 1
    emu.read = lambda addr, mt, signed: 0
    emu.getScreenBuffer = lambda: lua.table_from([0x808080] * (256 * 240), )  # 1-based, shifted; fine for a mock
    emu.setInput = lambda port, btns: state["inputs"].append(dict(btns.items()) if hasattr(btns, "items") else None)
    emu.getInput = lambda port: lua.table()
    emu.createSavestate = lambda: "state"
    emu.loadSavestate = lambda s: None
    emu.stop = lambda code=0: state.__setitem__("stopped", code)
    # buffer needs index 0 too:
    lua.execute("""
      local orig = emu.getScreenBuffer
      emu.getScreenBuffer = function() local t = orig(); t[0] = 0x808080; return t end
    """)
    err = lua.eval("function(src) local f, e = load(src, '@bridge'); if not f then error(e) end; f() end")
    err(BRIDGE)

    def frame():
        cbs["inputPolled"]()
        state["frames"] += 1
        cbs["endFrame"]()   # may block (lock-step)

    def send(token, action):
        tmp = ipc / "host.tmp"; tmp.write_text(f"A {token} {action}\n")
        os.replace(tmp, ipc / "command.in")

    def obs_exists(token):
        return (ipc / f"observation_ready_{token}.ready").exists()

    log = []
    # Host thread: waits for observation, "thinks" for host_delays[i] seconds, replies.
    def host():
        send("t0", 0)  # first command (initial observation request)
        for i, d in enumerate(host_delays):
            tok = f"t{i}"
            while not obs_exists(tok): time.sleep(0.001)
            log.append((tok, state["frames"], time.time()))
            time.sleep(d)
            send(f"t{i+1}", 0)
        tok = f"t{len(host_delays)}"
        while not obs_exists(tok): time.sleep(0.001)
        log.append((tok, state["frames"], time.time()))
        state["done"] = True

    state["done"] = False
    th = threading.Thread(target=host); th.start()
    t_start = time.time()
    # In free-run mode there is no blocking, so emulate "unthrottled" by stepping frames until done.
    while not state["done"] and time.time() - t_start < 30:
        frame()
        if not lockstep:
            pass
    th.join()
    shutil.rmtree(ipc, ignore_errors=True)
    return log

for lockstep in (True, False):
    print(f"--- lockstep={lockstep}")
    log = run(lockstep, host_delays=[0.0, 0.3, 1.5, 0.05])
    prev = None
    for tok, frames, ts in log:
        print(f"obs {tok}: emulator frame {frames}" + (f"  (+{frames-prev} frames since previous obs)" if prev is not None else ""))
        prev = frames
