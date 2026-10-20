-- Stand-in for the ORIGINAL Mesen (0.9.x) Lua API, matching the build whose api_info.txt
-- the user posted:
--   * emu.eventType / emu.memCallbackType are numeric; emu.callbackType and
--     emu.memType.nesMemory do not exist; emu.createSavestate does not exist.
--   * emu.saveSavestate() returns the state string; emu.loadSavestate(string) -> bool.
--     The current bridge must call them inside a CPU-exec memory callback; the mock
--     also permits startFrame for exercising the deliberately incompatible legacy
--     fixture. Current bridge tests model Mesen 2's `emu.callbackType.exec` API.
--   * A script callback that runs longer than mock.timeout seconds is aborted with
--     "Maximum execution time (...) exceeded." (Mesen default: 1000 ms, per callback).
mock = {frame = 0, ram = {}, ev = {}, memory_callbacks = {}, next_ref = 1,
        in_startframe = false, in_exec = false, timeout = 1.0,
        timeout_hits = 0, callback_errors = {}, input_history = {}}
emu = {
  eventType = {reset = 0, nmi = 1, irq = 2, startFrame = 3, endFrame = 4, codeBreak = 5,
               stateLoaded = 6, stateSaved = 7, inputPolled = 8, scriptEnded = 10},
  memType = {cpu = 0, nesMemory = 0, ppu = 1, palette = 2, oam = 3, prgRom = 5, workRam = 8},
  callbackType = {read = 0, write = 1, exec = 2},
  memCallbackType = {cpuRead = 0, cpuWrite = 1, cpuExec = 2, ppuRead = 3, ppuWrite = 4},
}
function emu.addEventCallback(fn, t)
  assert(t ~= nil, "addEventCallback: nil event type")
  mock.ev[t] = mock.ev[t] or {}
  local ref = mock.next_ref; mock.next_ref = ref + 1
  table.insert(mock.ev[t], {fn = fn, ref = ref}); return ref
end
function emu.removeEventCallback(fn, t) end
function emu.getState() return {cpu = {pc = mock.pc or 0x8000}, frameCount = mock.frame} end
function emu.getCpuState() return {pc = mock.pc or 0x8000} end
function emu.addMemoryCallback(fn, t, start_addr, end_addr, cpu_type, memory_type)
  assert(type(fn) == "function", "addMemoryCallback: callback must be a function")
  assert(t ~= nil, "addMemoryCallback: nil callback type")
  local ref = mock.next_ref; mock.next_ref = ref + 1
  table.insert(mock.memory_callbacks, {fn = fn, type = t, start_addr = start_addr or 0,
      end_addr = end_addr or (start_addr or 0xFFFF), cpu_type = cpu_type or 0,
      memory_type = memory_type or 0, ref = ref})
  return ref
end
function emu.removeMemoryCallback(ref, t, start_addr, end_addr, cpu_type, memory_type)
  for i = #mock.memory_callbacks, 1, -1 do
    local cb = mock.memory_callbacks[i]
    if cb.ref == ref and cb.type == t then
      table.remove(mock.memory_callbacks, i)
      return
    end
  end
  error("removeMemoryCallback: callback ref/type not found")
end
function emu.read(addr, t, signed) return mock.ram[addr] or 0 end   -- nil type defaults to cpu memory
mock.obs_frames = {}
function emu.getScreenBuffer()
  mock.obs_frames[#mock.obs_frames + 1] = mock.frame   -- frame at which each observation is captured
  local t = {}; for i = 0, 256*240 do t[i] = 0x808080 end; return t
end
-- Modern MesenCE 2.x is setInput(input, port, subPort); the intentionally
-- legacy original_bridge.lua fixture still uses setInput(port, input). Support
-- both in the shared mock, while source tests require the production bridge to
-- use the modern order.
function emu.setInput(first, second, third)
  local input, port, subport
  if type(first) == "table" then
    input, port, subport = first, second or 0, third or 0
  else
    port, input, subport = first or 0, second, third or 0
  end
  assert(type(input) == "table", "emu.setInput expects an input table")
  local copy = {}
  for k, v in pairs(input) do copy[k] = v end
  mock.last_input = copy
  mock.input_history[#mock.input_history + 1] = {
    frame = mock.frame, port = port, subport = subport, buttons = copy
  }
end
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
local function fire_exec()
  mock.in_exec = true
  -- Match the one-shot behavior closely enough for bridge savestate tests: one
  -- main-CPU instruction dispatch occurs after startFrame and before endFrame.
  local current = {}
  for _, cb in ipairs(mock.memory_callbacks) do
    if cb.type == (emu.callbackType and emu.callbackType.exec or emu.memCallbackType.cpuExec) then
      current[#current + 1] = cb
    end
  end
  local address = mock.pc or 0x8000
  for _, cb in ipairs(current) do
    local still_active = false
    for _, active in ipairs(mock.memory_callbacks) do
      if active.ref == cb.ref then still_active = true; break end
    end
    local in_range = address >= cb.start_addr and address <= cb.end_addr
    if still_active and in_range then call_cb(cb, address, 0) end
  end
  mock.in_exec = false
end
function mock.run_frame()
  fire(emu.eventType.startFrame, "in_startframe")
  fire_exec()
  fire(emu.eventType.inputPolled)
  mock.frame = mock.frame + 1
  mock.frames_run = (mock.frames_run or 0) + 1
  fire(emu.eventType.endFrame)
end
function mock.active_callbacks() return #mock.memory_callbacks end
