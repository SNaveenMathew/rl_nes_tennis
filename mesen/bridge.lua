-- Tennis RL bridge for Mesen / MesenCE.
--
-- Transport: atomic file mailbox. No LuaSocket or network access is required.
--
-- Worker directory comes from MESEN_RL_IPC_DIR and contains:
--   command.in       latest host -> Mesen command (bounded, one pending command)
--   observation_<token>.bin  Mesen -> host observation
--   ready.txt        bridge startup marker
--
-- Commands:
--   A <action>   latch controller action
--   R            reset emulator
--   Q            quit
--   SV           create in-memory savestate
--   LS           load in-memory savestate
--   PING         write pong.txt
--
-- Observation:
--   O <seq> <reward> <done> <player_score> <player_games> <point_end>\n
--   followed by WIDTH*HEIGHT raw grayscale bytes; token is host-generated.
--
-- The policy receives only video. RAM is used only for reward diagnostics/reset.

local IPC_DIR = os.getenv("MESEN_RL_IPC_DIR") or "runs/mesen_ipc/worker"
local WIDTH = tonumber(os.getenv("MESEN_RL_W") or "84")
local HEIGHT = tonumber(os.getenv("MESEN_RL_H") or "84")
local FRAME_SKIP = tonumber(os.getenv("MESEN_RL_FRAME_SKIP") or "4")
local GAME_LEVEL = tonumber(os.getenv("MESEN_RL_LEVEL") or "1")
local AUTO_BOOT = os.getenv("MESEN_RL_AUTO_BOOT") ~= "0"
-- NES renders a fixed 256x240 framebuffer for this game.
local SRC_W = 256
local SRC_H = 240
local MATCH_GAMES = tonumber(os.getenv("MESEN_RL_MATCH_GAMES") or "6")
local REWARD_POINT_WIN = tonumber(os.getenv("MESEN_RL_POINT_WIN") or "1.0")
local REWARD_POINT_LOSS = tonumber(os.getenv("MESEN_RL_POINT_LOSS") or "-1.0")
local REWARD_GAME_WIN = tonumber(os.getenv("MESEN_RL_GAME_WIN") or "5.0")

local COMMAND_PATH = IPC_DIR .. "/command.in"
local OBS_PREFIX = IPC_DIR .. "/observation_"
local OBS_READY_PREFIX = IPC_DIR .. "/observation_ready_"
local READY_PATH = IPC_DIR .. "/ready.txt"
local ERROR_PATH = IPC_DIR .. "/error.txt"

local RAM_PLAYER_SCORE = 0x0058
local RAM_OPPONENT_SCORE = 0x0059
local RAM_PLAYER_GAMES = 0x005A
local RAM_OPPONENT_GAMES = 0x005B
local RAM_POINT_END = 0x0072

local current_action = 0
local frame_counter = 0
local step_pending = false
local pending_token = nil
local initialized = false
local reset_pending = false
local last_player_score = 0
local last_player_games = 0
local last_point_end = 0
local saved_state = nil
local match_savestate = nil
local boot_frame = 0
local boot_complete = not AUTO_BOOT
local save_pending = false
local load_pending = false
local stopped = false
local held_buttons = {}  -- button_name -> frames_remaining; used by HOLD/START commands

local BUTTONS = {
    left = false,
    right = false,
    up = false,
    down = false,
    a = false,
    b = false,
    start = false,
    select = false,
}

local ACTIONS = {
    {},
    {"left"},
    {"right"},
    {"up"},
    {"down"},
    {"a"},
    {"b"},
    {"left", "a"},
    {"right", "a"},
    {"up", "a"},
    {"down", "a"},
    {"left", "b"},
    {"right", "b"},
    {"up", "b"},
    {"down", "b"},
    {"left", "up"},
    {"right", "up"},
    {"left", "down"},
    {"right", "down"},
}

local function write_text(path, text)
    local f = io.open(path, "wb")
    if not f then return false end
    f:write(text)
    f:close()
    return true
end

local function atomic_write(path, tmp, data)
    local f = io.open(tmp, "wb")
    if not f then return false end
    f:write(data)
    f:flush()
    f:close()
    os.remove(path)
    local ok = os.rename(tmp, path)
    return ok ~= nil
end

local function report_error(text)
    pcall(function() write_text(ERROR_PATH, text .. "\n") end)
end

local function clear_buttons()
    for k, _ in pairs(BUTTONS) do BUTTONS[k] = false end
end

local function read_command()
    local f = io.open(COMMAND_PATH, "rb")
    if not f then return nil end
    local line = f:read("*l")
    f:close()
    os.remove(COMMAND_PATH)
    return line
end

local function apply_command(line)
    if not line then return false end
    local cmd, rest = line:match("^(%S+)%s*(.*)$")
    if cmd == "A" then
        local token, action_text = rest:match("^(%S+)%s+(%S+)$")
        local action = tonumber(action_text)
        if token and action and ACTIONS[action + 1] then
            current_action = action
            pending_token = token
            step_pending = true
            frame_counter = 0
        end
    elseif cmd == "R" then
        if match_savestate ~= nil then
            pcall(function()
                emu.loadSavestate(match_savestate)
            end)
            initialized = false
            frame_counter = 0
            step_pending = false
            pending_token = nil
            current_action = 0
        else
            report_error("R command is unsupported without match savestate; restart the worker for reset")
            return false
        end
    elseif cmd == "SV" then
        save_pending = true
    elseif cmd == "LS" then
        load_pending = true
    elseif cmd == "HOLD" then
        -- HOLD button_name N_frames  — hold any NES button for N frames.
        -- Valid button names: left, right, up, down, a, b, start, select.
        local button_name, n_text = rest:match("^(%S+)%s+(%S+)$")
        local n = tonumber(n_text) or 0
        if button_name and n > 0 and BUTTONS[button_name] ~= nil then
            held_buttons[button_name] = math.max(1, math.floor(n))
        end
    elseif cmd == "START" then
        -- Backward-compat alias: START N  ≡  HOLD start N
        local n = tonumber(rest) or 60
        held_buttons["start"] = math.max(1, math.floor(n))
    elseif cmd == "Q" then
        stopped = true
        pcall(function() write_text(READY_PATH, "BYE\n") end)
        emu.stop(0)
    elseif cmd == "PING" then
        write_text(IPC_DIR .. "/pong.txt", "PONG\n")
    end
    return true
end

local function poll_command()
    -- Controller/reset input is a latest-value mailbox. There is never more
    -- than one pending command, so Mesen cannot fall behind a durable queue.
    local line = read_command()
    if line then
        apply_command(line)
    end
end

local function get_boot_input()
    clear_buttons()
    boot_frame = boot_frame + 1

    -- Step 1: Wait for title screen to appear (60 frames)
    if boot_frame <= 60 then
        return BUTTONS
    -- Step 2: Press Start (15 frames) to select Singles (1 Player: 1 Game)
    elseif boot_frame <= 75 then
        BUTTONS["start"] = true
        return BUTTONS
    -- Step 3: Wait for Level Selection screen (40 frames)
    elseif boot_frame <= 115 then
        return BUTTONS
    end

    -- Step 4: Navigate to desired GAME_LEVEL (1 to 5)
    local cur_f = 116
    for step = 1, (GAME_LEVEL - 1) do
        if boot_frame >= cur_f and boot_frame < cur_f + 8 then
            BUTTONS["right"] = true
            return BUTTONS
        elseif boot_frame >= cur_f + 8 and boot_frame < cur_f + 20 then
            return BUTTONS
        end
        cur_f = cur_f + 20
    end

    -- Step 5: Press Start (15 frames) to confirm Level and start the match
    if boot_frame >= cur_f and boot_frame < cur_f + 15 then
        BUTTONS["start"] = true
        return BUTTONS
    end

    -- Step 6: Wait for court transition / players to take positions
    local finish_f = cur_f + 75
    if boot_frame >= finish_f and not boot_complete then
        boot_complete = true
        pcall(function()
            match_savestate = emu.createSavestate()
        end)
        local ok = write_text(READY_PATH, string.format("READY %d %d %d\n", WIDTH, HEIGHT, FRAME_SKIP))
        if not ok then
            report_error("Unable to initialize IPC mailbox; enable Mesen Lua I/O/OS access")
            emu.stop(32)
        end
    end

    return BUTTONS
end

local function current_input()
    if not boot_complete then
        return get_boot_input()
    end
    clear_buttons()
    -- Apply any button holds (HOLD/START commands override the policy action).
    local any_held = false
    for button, frames in pairs(held_buttons) do
        if frames > 0 then
            BUTTONS[button] = true
            any_held = true
        end
    end
    if any_held then
        return BUTTONS
    end
    local buttons = ACTIONS[current_action + 1] or ACTIONS[1]
    for _, name in ipairs(buttons) do BUTTONS[name] = true end
    return BUTTONS
end

local function read_u8(addr)
    return emu.read(addr, emu.memType.nesMemory, false)
end

local XMAP = {}
local YMAP = {}
for x = 0, WIDTH - 1 do
    XMAP[x + 1] = math.floor((x + 0.5) * SRC_W / WIDTH)
end
for y = 0, HEIGHT - 1 do
    YMAP[y + 1] = math.floor((y + 0.5) * SRC_H / HEIGHT)
end

local function capture_frame_gray()
    local buffer = emu.getScreenBuffer()
    local out = {}
    local n = 0
    for y = 0, HEIGHT - 1 do
        local sy = YMAP[y + 1]
        local row = sy * SRC_W
        for x = 0, WIDTH - 1 do
            local sx = XMAP[x + 1]
            local argb = buffer[row + sx]
            if argb == nil then
                error(string.format("screen buffer nil at x=%d y=%d idx=%d", x, y, row + sx))
            end
            local r = math.floor(argb / 0x10000) % 0x100
            local g = math.floor(argb / 0x100) % 0x100
            local b = argb % 0x100
            n = n + 1
            out[n] = string.char(math.floor(0.299 * r + 0.587 * g + 0.114 * b + 0.5))
        end
    end
    return table.concat(out)
end

local function compute_reward()
    local player_score = read_u8(RAM_PLAYER_SCORE)
    local player_games = read_u8(RAM_PLAYER_GAMES)
    local opponent_games = read_u8(RAM_OPPONENT_GAMES)
    local point_end = read_u8(RAM_POINT_END)

    if not initialized then
        last_player_score = player_score
        last_player_games = player_games
        last_point_end = point_end
        initialized = true
        return 0.0, false, player_score, player_games, point_end
    end

    local reward = 0.0
    if player_games > last_player_games then
        reward = reward + REWARD_GAME_WIN + REWARD_POINT_WIN
    elseif point_end ~= 0 and last_point_end == 0 then
        if player_score > last_player_score then
            reward = reward + REWARD_POINT_WIN
        else
            reward = reward + REWARD_POINT_LOSS
        end
    end

    -- Episode ends when either player wins MATCH_GAMES games.  Previously only
    -- the player's count was checked, leaving the emulator running indefinitely
    -- through the opponent-win attract sequence.
    local done = player_games >= MATCH_GAMES or opponent_games >= MATCH_GAMES
    last_player_score = player_score
    last_player_games = player_games
    last_point_end = point_end
    return reward, done, player_score, player_games, point_end
end

local function send_observation()
    local frame = capture_frame_gray()
    local reward, done, player_score, player_games, point_end = compute_reward()
    if pending_token == nil then
        report_error("Observation requested without a pending host token")
        emu.stop(35)
        return
    end
    local token = pending_token
    local header = string.format(
        "O %s %.6f %d %d %d %d\n",
        token, reward, done and 1 or 0, player_score, player_games, point_end
    )
    local suffix = token
    local obs_path = OBS_PREFIX .. suffix .. ".bin"
    local obs_tmp = OBS_PREFIX .. suffix .. ".tmp"
    local ready_path = OBS_READY_PREFIX .. suffix .. ".ready"
    local ready_tmp = OBS_READY_PREFIX .. suffix .. ".tmp"
    os.remove(ready_path)
    os.remove(obs_path)
    if not atomic_write(obs_path, obs_tmp, header .. frame) then
        report_error("Failed to write observation mailbox: " .. obs_path)
        emu.stop(31)
        return
    end
    if not atomic_write(ready_path, ready_tmp, "READY\n") then
        report_error("Failed to publish observation readiness: " .. ready_path)
        emu.stop(31)
    end
end

local input_callback = emu.addEventCallback(function()
    -- Action is consumed at every controller poll, exactly where Mesen recommends
    -- setInput() be applied.
    emu.setInput(0, current_input())
end, emu.eventType.inputPolled)

local function process_state_requests()
    if save_pending then
        local ok = pcall(function()
            saved_state = emu.createSavestate()
        end)
        save_pending = false
        if ok and saved_state ~= nil then
            write_text(IPC_DIR .. "/state_saved.txt", "SAVED\n")
        else
            report_error("Failed to create savestate")
        end
    end
    if load_pending and saved_state ~= nil then
        local ok = pcall(function()
            emu.loadSavestate(saved_state)
        end)
        load_pending = false
        initialized = false
        frame_counter = 0
        if ok then
            write_text(IPC_DIR .. "/state_loaded.txt", "LOADED\n")
        else
            report_error("Failed to load savestate")
        end
    end
end

local end_callback = emu.addEventCallback(function()
    if stopped then return end

    local ok, err = pcall(function()
        if not boot_complete then
            return
        end

        -- Poll commands every emulator frame. This keeps reset/action latency
        -- deterministic under --testrunner; the expensive operation (frame
        -- serialization) remains limited to FRAME_SKIP frames below.
        poll_command()

        process_state_requests()

        -- Count down held-button frames (HOLD/START commands from watch.py).
        for button, frames in pairs(held_buttons) do
            if frames > 0 then
                held_buttons[button] = frames - 1
            end
        end

        -- Lock-step environment protocol: Python publishes one command, and
        -- Mesen emits exactly one observation after FRAME_SKIP frames. This
        -- prevents observations from outrunning the learner and eliminates
        -- unbounded observation-file backlogs.
        if step_pending then
            frame_counter = frame_counter + 1
            if frame_counter >= FRAME_SKIP then
                send_observation()
                step_pending = false
                pending_token = nil
                frame_counter = 0
            end
        end
    end)
    if not ok then
        report_error("endFrame callback exception: " .. tostring(err))
        emu.stop(34)
    end
end, emu.eventType.endFrame)

local function startup()
    os.remove(ERROR_PATH)
    os.remove(READY_PATH)
    if not AUTO_BOOT then
        local ok = write_text(READY_PATH, string.format("READY %d %d %d\n", WIDTH, HEIGHT, FRAME_SKIP))
        if not ok then
            report_error("Unable to initialize IPC mailbox; enable Mesen Lua I/O/OS access")
            emu.stop(32)
        end
    end
end

local ok, err = pcall(startup)
if not ok then
    report_error("Bridge startup exception: " .. tostring(err))
    emu.stop(33)
end

-- No emu.step() loop is used: --testrunner drives the emulator, callbacks above
-- provide frame synchronization and emu.stop() terminates the runner.
