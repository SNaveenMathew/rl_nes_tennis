-- Stand-in for the ORIGINAL Mesen (0.9.x) Lua API, matching the build whose api_info.txt
-- the user posted:
--   * emu.eventType / emu.memCallbackType are numeric; emu.callbackType and
--     emu.memType.nesMemory do not exist; emu.createSavestate does not exist.
--   * emu.saveSavestate() returns the state string; emu.loadSavestate(string) -> bool.
--     Both only work inside a startFrame event callback or a cpuExec memory callback.
--   * A script callback that runs longer than mock.timeout seconds is aborted with
--     "Maximum execution time (...) exceeded." (Mesen default: 1000 ms, per callback).
mock = {frame = 0, ram = {}, ev = {}, next_ref = 1, in_startframe = false, in_exec = false,
        timeout = 1.0, timeout_hits = 0, callback_errors = {}}
emu = {
  eventType = {reset = 0, nmi = 1, irq = 2, startFrame = 3, endFrame = 4, codeBreak = 5,
               stateLoaded = 6, stateSaved = 7, inputPolled = 8, scriptEnded = 10},
  memType = {cpu = 0, ppu = 1, palette = 2, oam = 3, prgRom = 5, workRam = 8},
  memCallbackType = {cpuRead = 0, cpuWrite = 1, cpuExec = 2, ppuRead = 3, ppuWrite = 4},
}
function emu.addEventCallback(fn, t)
  assert(t ~= nil, "addEventCallback: nil event type")
  mock.ev[t] = mock.ev[t] or {}
  local ref = mock.next_ref; mock.next_ref = ref + 1
  table.insert(mock.ev[t], {fn = fn, ref = ref}); return ref
end
function emu.removeEventCallback(fn, t) end
function emu.addMemoryCallback() return 0 end
function emu.removeMemoryCallback() end
function emu.read(addr, t, signed) return mock.ram[addr] or 0 end   -- nil type defaults to cpu memory
mock.obs_frames = {}
function emu.getScreenBuffer()
  mock.obs_frames[#mock.obs_frames + 1] = mock.frame   -- frame at which each observation is captured
  local t = {}; for i = 0, 256*240 do t[i] = 0x808080 end; return t
end
function emu.setInput(port, b) mock.last_input = b end
function emu.getInput(port) return {} end
function emu.stop(code) mock.stopped = code end
local function check_allowed()
  if not (mock.in_startframe or mock.in_exec or mock.savestate_anywhere) then
    error("This function must be called inside a StartFrame event callback or a CpuExec memory operation callback")
  end
end
function emu.saveSavestate()
  check_allowed()
  local parts = {"STATE", mock.frame}
  for _, a in ipairs({0x58, 0x59}) do parts[#parts + 1] = mock.ram[a] or 0 end
  return table.concat(parts, "|") .. "|\0binary\0tail"          -- embedded NULs must survive
end
function emu.loadSavestate(s)
  check_allowed()
  local f, p, o = s:match("^STATE|(%d+)|(%d+)|(%d+)|\0binary\0tail$")
  if not f then return false end
  mock.frame = tonumber(f); mock.ram[0x58] = tonumber(p); mock.ram[0x59] = tonumber(o)
  mock.loaded = (mock.loaded or 0) + 1
  return true
end
-- Run one callback the way Mesen does: timer reset, 1000-instruction count hook, errors logged.
local function call_cb(c, ...)
  local t0 = os.clock()
  debug.sethook(function()
    if os.clock() - t0 > mock.timeout then
      mock.timeout_hits = mock.timeout_hits + 1
      error("Maximum execution time (" .. math.floor(mock.timeout * 1000) .. " ms) exceeded.")
    end
  end, "", 1000)
  local ok, err = pcall(c.fn, ...)
  debug.sethook()
  if not ok then table.insert(mock.callback_errors, tostring(err)) end
end
local function fire(t, flag)
  if flag then mock[flag] = true end
  for _, c in ipairs(mock.ev[t] or {}) do call_cb(c) end
  if flag then mock[flag] = false end
end
function mock.run_frame()
  fire(emu.eventType.startFrame, "in_startframe")
  fire(emu.eventType.inputPolled)
  mock.frame = mock.frame + 1
  mock.frames_run = (mock.frames_run or 0) + 1
  fire(emu.eventType.endFrame)
end
function mock.active_callbacks() return 0 end
