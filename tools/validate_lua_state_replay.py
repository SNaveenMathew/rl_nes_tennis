import ctypes, ctypes.util, os, tempfile, threading, time, shutil
from pathlib import Path
LIB = ctypes.util.find_library('lua5.4')
if not LIB: raise SystemExit('liblua5.4 not found')
lua = ctypes.CDLL(LIB); P = ctypes.c_void_p
lua.luaL_newstate.restype=P; lua.luaL_openlibs.argtypes=[P]
lua.luaL_loadfilex.argtypes=[P,ctypes.c_char_p,ctypes.c_char_p]; lua.luaL_loadfilex.restype=ctypes.c_int
lua.lua_pcallk.argtypes=[P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_longlong,ctypes.c_void_p]; lua.lua_pcallk.restype=ctypes.c_int
lua.lua_getglobal.argtypes=[P,ctypes.c_char_p]; lua.lua_getfield.argtypes=[P,ctypes.c_int,ctypes.c_char_p]; lua.lua_settop.argtypes=[P,ctypes.c_int]
lua.lua_tointegerx.argtypes=[P,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]; lua.lua_tointegerx.restype=ctypes.c_longlong
lua.lua_tolstring.argtypes=[P,ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]; lua.lua_tolstring.restype=ctypes.c_char_p
ROOT=Path(__file__).resolve().parents[1]
def check(L, rc, label):
    if rc:
        n=ctypes.c_size_t(); msg=lua.lua_tolstring(L,-1,ctypes.byref(n)); lua.lua_settop(L,0)
        raise RuntimeError(f'{label}: {msg[:n.value].decode(errors="replace") if msg else rc}')
    lua.lua_settop(L,0)
def execfile(L,p):
    rc=lua.luaL_loadfilex(L,str(p).encode(),None)
    if rc:
        check(L,rc,'load '+str(p))
    rc=lua.lua_pcallk(L,0,0,0,0,None); check(L,rc,'execute '+str(p))
def getint(L, field):
    lua.lua_getglobal(L,b'mock'); lua.lua_getfield(L,-1,field.encode()); isnum=ctypes.c_int()
    val=lua.lua_tointegerx(L,-1,ctypes.byref(isnum)); lua.lua_settop(L,0)
    return int(val) if isnum.value else None
def runframe(L):
    lua.lua_getglobal(L,b'mock'); lua.lua_getfield(L,-1,b'run_frame')
    rc=lua.lua_pcallk(L,0,0,1,0,None); check(L,rc,'mock.run_frame')
def wait(ipc,name,timeout=5):
    t=time.time()
    while time.time()-t<timeout:
        if (ipc/name).exists(): return True
        time.sleep(.001)
    return False
def one_run():
    ipc=Path(tempfile.mkdtemp(prefix='lua_state_replay_'))
    os.environ.update({'MESEN_RL_IPC_DIR':str(ipc),'MESEN_RL_W':'8','MESEN_RL_H':'8','MESEN_RL_FRAME_SKIP':'4','MESEN_RL_AUTO_BOOT':'0','MESEN_RL_LOCKSTEP':'1','MESEN_RL_LOCKSTEP_TIMEOUT':'20','MESEN_RL_LOCKSTEP_SLICE':'0.25','MESEN_RL_STRICT_LOCKSTEP':'1','MESEN_RL_STRICT_WAIT_S':'0.70'})
    L=lua.luaL_newstate(); lua.luaL_openlibs(L); execfile(L,ROOT/'tests/mock_emu.lua'); execfile(L,ROOT/'mesen/bridge.lua')
    state={'done':False}
    def send(line):
        tmp=ipc/'h.tmp'; tmp.write_text(line+'\n'); os.replace(tmp,ipc/'command.in')
    def step(tok):
        marker=ipc/f'observation_ready_{tok}.ready'; marker.unlink(missing_ok=True)
        send(f'A {tok} 0'); assert wait(ipc,f'observation_ready_{tok}.ready'),f'no obs {tok}'
    def snap(name):
        send(f'SAVE_TO {ipc}/{name}.state'); assert wait(ipc,'state_saved.txt'),f'no save ack {name}'
        (ipc/'state_saved.txt').unlink(); return int((ipc/f'{name}.state').read_text().split('|')[1])
    def host():
        try:
            for i in range(3): step(f'a{i}')
            base=snap('base'); loads=[]
            for k in range(3):
                send(f'LOAD_FROM 7 {ipc}/base.state'); assert wait(ipc,'state_loaded.txt'),'no load ack 7'
                (ipc/'state_loaded.txt').unlink(); loads.append(snap(f'l{k}'))
            send(f'LOAD_FROM 0 {ipc}/base.state'); assert wait(ipc,'state_loaded.txt'),'no load ack 0'
            (ipc/'state_loaded.txt').unlink(); zero=snap('zero')
            gaps=[]; prev=snap('p0')
            for i in range(4):
                time.sleep(.15); step(f's{i}'); cur=snap(f'p{i+1}'); gaps.append(cur-prev); prev=cur
            for _ in range(3): step('idle')
            # mock.active_callbacks() is intentionally zero in mock_emu.lua.
            state.update(base=base,loads=loads,zero=zero,gaps=gaps,armed_when_idle=0)
        except BaseException as e: state['error']=repr(e)
        finally: state['done']=True
    thread=threading.Thread(target=host,daemon=True); thread.start(); start=time.time()
    while not state['done'] and time.time()-start<30: runframe(L)
    thread.join(1)
    if not state['done']: state['error']='host timeout'
    state['frame']=getint(L,'frame'); state['loaded']=getint(L,'loaded')
    lua.lua_close.argtypes=[P]; lua.lua_close(L); shutil.rmtree(ipc,ignore_errors=True)
    return state
for i in range(20):
    r=one_run(); print(i+1,r)
    if 'error' in r: raise SystemExit(r['error'])
    assert len(set(r['loads']))==1, r
    assert r['loads'][0]==r['base']+7, r
    assert r['zero']==r['base']+1, r
    assert len(set(r['gaps']))==1, r
print('PASS: 20 full Lua 5.4 ctypes state-op/determinism runs passed')
