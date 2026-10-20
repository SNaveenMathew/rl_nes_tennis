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
--   O <token> <reward> <done> <player_score> <player_games>\n
--     <opponent_score> <opponent_games> <point_end>\n
--   followed by WIDTH*HEIGHT raw grayscale bytes; in interactive recording with
--   native capture enabled, 256*240*3 raw RGB bytes follow the grayscale payload.
--   Token is host-generated.
--
-- The policy receives only video. RAM is used only for reward diagnostics/reset.

-- Mesen's Windows Lua I/O can reject mixed-separator absolute paths on some builds.
-- Canonicalize all bridge filesystem paths to forward slashes (accepted by Windows,
-- Linux and macOS) before deriving the mailbox file names.
local IPC_DIR = (os.getenv("MESEN_RL_IPC_DIR") or "runs/mesen_ipc/worker"):gsub("\\", "/")

local function normalize_file_path(path)
    -- Host paths may be produced by pathlib on Windows (backslashes), by shell
    -- scripts (mixed separators), or by POSIX tools (forward slashes).
    -- Keep the drive/UNC prefix intact while standardizing separators.
    return tostring(path or ""):gsub("\\", "/")
end
local WIDTH = tonumber(os.getenv("MESEN_RL_W") or "128")
local HEIGHT = tonumber(os.getenv("MESEN_RL_H") or "120")
local FRAME_SKIP = tonumber(os.getenv("MESEN_RL_FRAME_SKIP") or "2")
local GAME_LEVEL = tonumber(os.getenv("MESEN_RL_LEVEL") or "1")
local RECORD_MODE = os.getenv("MESEN_RL_RECORD_MODE") == "1"
-- Expensive native RGB capture is enabled only for human demonstration recording.
local CAPTURE_NATIVE_FRAMES = RECORD_MODE and os.getenv("MESEN_RL_CAPTURE_NATIVE") == "1"
local AUTO_BOOT = os.getenv("MESEN_RL_AUTO_BOOT") ~= "0"
-- Lock-step mode: after each observation is published the emulator WAITS (inside
-- the endFrame callback) for the host's next command, so the reply takes effect on
-- the very next frame instead of after a random number of free-running frames.
-- Mesen aborts any script callback that runs longer than its script timeout (1000 ms
-- by default, not raisable from the command line), so the wait is done in short
-- slices: after MESEN_RL_LOCKSTEP_SLICE seconds the callback returns and the game
-- advances ONE frame with no input, then waits again. A slow learner update (seconds)
-- therefore costs a handful of idle frames instead of killing the bridge.
-- Off in record mode: the human has to keep playing in real time.
local LOCKSTEP = (os.getenv("MESEN_RL_LOCKSTEP") ~= "0") and not RECORD_MODE
local LOCKSTEP_SLICE_S = tonumber(os.getenv("MESEN_RL_LOCKSTEP_SLICE") or "0.25")
-- An orphaned emulator (host crashed) stops itself after this many silent seconds.
local LOCKSTEP_TIMEOUT_S = tonumber(os.getenv("MESEN_RL_LOCKSTEP_TIMEOUT") or "600")
-- Strict lockstep is for deterministic replay/evaluation. Unlike normal training,
-- it fails closed if the host does not submit its next command within the deadline,
-- rather than allowing a timeout-driven idle frame to change the evaluated state.
-- Keep the deadline below Mesen's Lua callback watchdog (typically about 1 second).
local STRICT_LOCKSTEP = os.getenv("MESEN_RL_STRICT_LOCKSTEP") == "1"
local STRICT_LOCKSTEP_WAIT_S = tonumber(os.getenv("MESEN_RL_STRICT_WAIT_S") or "0.70")
-- NES renders a fixed 256x240 framebuffer for this game.
local SRC_W = 256
local SRC_H = 240
local MATCH_GAMES = tonumber(os.getenv("MESEN_RL_MATCH_GAMES") or "6")
local REWARD_POINT_WIN = tonumber(os.getenv("MESEN_RL_POINT_WIN") or "1.0")
local REWARD_POINT_LOSS = tonumber(os.getenv("MESEN_RL_POINT_LOSS") or "-1.0")
local REWARD_GAME_WIN = tonumber(os.getenv("MESEN_RL_GAME_WIN") or "5.0")
local REWARD_GAME_LOSS = tonumber(os.getenv("MESEN_RL_GAME_LOSS") or "-5.0")
-- Match-level terminal bonus/penalty, applied on top of the final game's
-- REWARD_GAME_WIN/LOSS the moment either side reaches MATCH_GAMES. Gives the
-- agent a clearly bigger, distinct signal for closing out a whole match
-- rather than just accumulating individual game rewards.
local REWARD_MATCH_WIN = tonumber(os.getenv("MESEN_RL_MATCH_WIN") or "25.0")
local REWARD_MATCH_LOSS = tonumber(os.getenv("MESEN_RL_MATCH_LOSS") or "-10.0")
-- Stall watchdog: NES Tennis shows "point"/"game"/"advantage" overlay screens
-- (and possibly other prompts) that require a Start press to dismiss before
-- play continues. The RL action space intentionally has no Start button
-- (menu-advancing isn't a tennis skill), so without this the episode would
-- otherwise sit dead on such a screen — collecting zero reward — until the
-- 15000-step episode timeout. If the point score hasn't moved for this many
-- consecutive emulator frames, the bridge auto-taps Start on the agent's
-- behalf. Tune via MESEN_RL_STALL_FRAMES if 5s (300 frames) is too
-- eager/lazy for a real rally.
local STALL_FRAMES_BEFORE_START = tonumber(os.getenv("MESEN_RL_STALL_FRAMES") or "300")
local STALL_START_HOLD_FRAMES = 12

local COMMAND_PATH = IPC_DIR .. "/command.in"
local OBS_PREFIX = IPC_DIR .. "/observation_"
local OBS_READY_PREFIX = IPC_DIR .. "/observation_ready_"
local READY_PATH = IPC_DIR .. "/ready.txt"
local ERROR_PATH = IPC_DIR .. "/error.txt"

local RAM_PLAYER_SCORE = 0x0058
local RAM_OPPONENT_SCORE = 0x0059
local RAM_PLAYER_GAMES = 0x005B
local RAM_OPPONENT_GAMES = 0x005A
local RAM_POINT_END = 0x0072

local current_action = 0
local current_pulse_frames = -1 -- -1 = legacy button hold; positive = pulse A/B for N NES frames
local current_pulse_remaining = 0
local frame_counter = 0
local step_pending = false
local pending_token = nil
local initialized = false
local reset_pending = false
local last_player_score = 0
local last_opponent_score = 0
local last_point_end = 0
-- Game wins counted locally via score-reset detection (not from RAM counters,
-- which are not reliable incremental values in this ROM).
local actual_player_games = 0
local actual_opponent_games = 0
-- Per-frame sticky peak scores: updated every frame so we never miss the
-- brief "Advantage" frame (score=4) between FRAME_SKIP samples.
local max_player_score_seen = 0
local max_opponent_score_seen = 0
-- Stall watchdog state (see STALL_FRAMES_BEFORE_START above).
local frames_since_progress = 0
local last_frame_player_score = -1
local last_frame_opponent_score = -1
local stall_rescue_count = 0
local saved_state = nil
local match_savestate = nil
local boot_frame = 0
local boot_complete = not AUTO_BOOT
local save_pending = false
local load_pending = false
local stopped = false
local held_buttons = {}  -- button_name -> frames_remaining; used by HOLD/START commands
local last_host_activity = os.time()
-- Savestate machinery. Mesen 2's createSavestate/loadSavestate APIs must run from
-- a CPU exec memory callback (not startFrame/endFrame). startFrame only arms a
-- one-shot exec callback when work is queued, avoiding per-instruction overhead
-- during normal training. Older API names are supported through the same path.
local pending_op = nil          -- {kind="save"|"load"|"reset"|"save_to"|"load_from", path=, noop=}
local state_op_callback_id = nil
local state_op_callback_fired = false
local state_op_callback_address = nil
local EXEC_CALLBACK_TYPE = (emu.callbackType and emu.callbackType.exec)
    or (emu.memCallbackType and (emu.memCallbackType.cpuExec or emu.memCallbackType.exec))
local snapshot_pending = false  -- take the post-boot "match start" snapshot
local ready_written = false
local boot_complete_frames = 0
local noop_remaining = 0        -- free-running no-op frames after LOAD_FROM
local noop_ack_pending = false
local state_op_wait = 0
local screen_capture_wait_frames = 0
local human_action = 0
local interval_action = 0
local interval_a_frames = 0
local interval_b_frames = 0

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
    {}, {"a"}, {"b"},
    {"left"}, {"left", "a"}, {"left", "b"},
    {"right"}, {"right", "a"}, {"right", "b"},
    {"up"}, {"up", "a"}, {"up", "b"},
    {"down"}, {"down", "a"}, {"down", "b"},
    {"left", "up"}, {"left", "up", "a"}, {"left", "up", "b"},
    {"right", "up"}, {"right", "up", "a"}, {"right", "up", "b"},
    {"left", "down"}, {"left", "down", "a"}, {"left", "down", "b"},
    {"right", "down"}, {"right", "down", "a"}, {"right", "down", "b"},
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
    if line then
        -- Some Windows-side producers use text-mode write_text(), which emits
        -- CRLF. This file is intentionally opened in binary mode, so Lua may
        -- leave the carriage return attached to the final token/path unless we
        -- remove it explicitly. A trailing CR makes SAVE_TO/LOAD_FROM paths
        -- invalid on Windows while ordinary token commands appear to work.
        line = line:gsub("\r$", "")
        last_host_activity = os.time()
    end
    return line
end

local function apply_command(line)
    if not line then return false end
    local cmd, rest = line:match("^(%S+)%s*(.*)$")
    if cmd == "A" then
        local token, action_text, pulse_text = rest:match("^(%S+)%s+(%S+)%s*(%S*)$")
        local action = tonumber(action_text)
        local pulse = nil
        if pulse_text and pulse_text ~= "" then pulse = tonumber(pulse_text) end
        if token and action and ACTIONS[action + 1]
            and (pulse == nil or (pulse >= 1 and pulse <= FRAME_SKIP)) then
            current_action = action
            current_pulse_frames = pulse or -1
            current_pulse_remaining = (pulse and pulse > 0) and pulse or 0
            pending_token = token
            step_pending = true
            frame_counter = 0
            screen_capture_wait_frames = 0
        elseif token and action and ACTIONS[action + 1] then
            report_error("Invalid A/B pulse frame count; expected 1..FRAME_SKIP")
        end
    elseif cmd == "R" then
        -- Reset to the post-boot match start (restored at the next startFrame).
        if match_savestate ~= nil then
            pending_op = {kind = "reset"}
        else
            report_error("R command is unsupported without match savestate; restart the worker for reset")
            return false
        end
    elseif cmd == "SV" then
        pending_op = {kind = "save"}
    elseif cmd == "LS" then
        pending_op = {kind = "load"}
    elseif cmd == "SAVE_TO" then
        -- SAVE_TO <path>: write the current emulator state to a file.
        if rest ~= "" then pending_op = {kind = "save_to", path = normalize_file_path(rest)} end
    elseif cmd == "LOAD_FROM" then
        -- LOAD_FROM <noop_frames> <path>: load a state file, then let N frames run
        -- with no input (so one saved state is not one fixed trajectory).
        local n, path = rest:match("^(%d+)%s+(.+)$")
        if n and path then pending_op = {kind = "load_from", path = normalize_file_path(path), noop = tonumber(n)} end
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
        snapshot_pending = true  -- READY is written once the snapshot exists
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
    for _, name in ipairs(buttons) do
        if name ~= "a" and name ~= "b" then
            BUTTONS[name] = true
        elseif current_pulse_frames < 0 or current_pulse_remaining > 0 then
            BUTTONS[name] = true
        end
    end
    return BUTTONS
end

local function read_u8(addr)
    return emu.read(addr, emu.memType.nesMemory or emu.memType.cpu or 0, false)
end

-- Mesen releases may expose the framebuffer only after the first rendered frame.
-- getScreenBuffer is also zero-based in the documented API and has shipped with
-- 239 visible rows on some builds. Detect either a zero- or one-based table and
-- require a complete visible image before sampling it. Return nil to defer output
-- rather than crashing the entire Lua bridge during emulator startup.
local function get_valid_screen_buffer()
    local ok, buffer = pcall(function() return emu.getScreenBuffer() end)
    if not ok or type(buffer) ~= "table" then return nil end

    local base = nil
    if buffer[0] ~= nil then
        base = 0
    elseif buffer[1] ~= nil and buffer[SRC_W * (SRC_H - 1)] ~= nil then
        -- Accommodate a 1-based Lua array if supplied by a compatible build.
        base = 1
    else
        return nil
    end

    -- The known Mesen 2 API returns 239 rows (0..238); some variants return 240.
    -- Validate the first and last pixel of each visible row used for resampling.
    local visible_h = 0
    for y = 0, SRC_H - 1 do
        local first = buffer[base + y * SRC_W]
        local last = buffer[base + y * SRC_W + (SRC_W - 1)]
        if first == nil or last == nil then break end
        visible_h = y + 1
    end
    if visible_h < 239 then return nil end
    return buffer, base, visible_h
end

local function capture_frame_gray()
    local buffer, base, source_h = get_valid_screen_buffer()
    if buffer == nil then return nil end
    local out = {}
    local n = 0
    for y = 0, HEIGHT - 1 do
        local sy0 = math.floor(y * source_h / HEIGHT)
        local sy1 = math.floor((y + 1) * source_h / HEIGHT)
        if sy1 <= sy0 then sy1 = sy0 + 1 end
        if sy1 > source_h then sy1 = source_h end
        for x = 0, WIDTH - 1 do
            local sx0 = math.floor(x * SRC_W / WIDTH)
            local sx1 = math.floor((x + 1) * SRC_W / WIDTH)
            if sx1 <= sx0 then sx1 = sx0 + 1 end
            if sx1 > SRC_W then sx1 = SRC_W end
            local sum = 0.0
            local count = 0
            for sy = sy0, sy1 - 1 do
                local row = base + sy * SRC_W
                for sx = sx0, sx1 - 1 do
                    local argb = buffer[row + sx]
                    if argb == nil then return nil end
                    local r = math.floor(argb / 0x10000) % 0x100
                    local g = math.floor(argb / 0x100) % 0x100
                    local b = argb % 0x100
                    sum = sum + (0.299 * r + 0.587 * g + 0.114 * b)
                    count = count + 1
                end
            end
            if count == 0 then return nil end
            n = n + 1
            out[n] = string.char(math.floor(sum / count + 0.5))
        end
    end
    return table.concat(out)
end

-- Retain an RGB frame for annotation only. The model still receives grayscale
-- observations. This protocol keeps a stable 256x240x3 payload: where Mesen exposes
-- only 239 raster rows, the final payload row duplicates the last visible row.
local function capture_frame_rgb()
    local buffer, base, source_h = get_valid_screen_buffer()
    if buffer == nil then return nil end
    local out = {}
    local n = 0
    for y = 0, SRC_H - 1 do
        local sy = math.min(y, source_h - 1)
        local row = base + sy * SRC_W
        for x = 0, SRC_W - 1 do
            local argb = buffer[row + x]
            if argb == nil then return nil end
            local r = math.floor(argb / 0x10000) % 0x100
            local g = math.floor(argb / 0x100) % 0x100
            local b = argb % 0x100
            n = n + 1
            out[n] = string.char(r, g, b)
        end
    end
    return table.concat(out)
end

local function compute_reward()
    local player_score  = read_u8(RAM_PLAYER_SCORE)
    local opponent_score = read_u8(RAM_OPPONENT_SCORE)
    local point_end     = read_u8(RAM_POINT_END)  -- kept for diagnostic header field

    if not initialized then
        -- Snapshot starting state; game counters always start at 0 here.
        last_player_score   = player_score
        last_opponent_score = opponent_score
        last_point_end      = point_end
        actual_player_games   = 0
        actual_opponent_games = 0
        -- Seed the sticky peaks so the first score increase is detected correctly.
        max_player_score_seen   = player_score
        max_opponent_score_seen = opponent_score
        initialized = true
        return 0.0, false, player_score, 0, opponent_score, 0, point_end
    end

    local reward = 0.0

    -- Detect game end: both point-scores reset to 0 after being non-zero.
    -- The ROM resets 0x0058/0x0059 to 0 when either player wins a game.
    -- We use the sticky max scores captured every frame (not just at sample
    -- boundaries) to reliably identify who held "Advantage" (score == 4)
    -- even if we skipped that frame.
    local game_ended = (player_score == 0 and opponent_score == 0
                        and (last_player_score > 0 or last_opponent_score > 0))

    if game_ended then
        -- Determine winner using the highest scores seen since the last game end.
        local player_won_game
        if max_player_score_seen == 4 then
            -- Player held Advantage → player won.
            player_won_game = true
        elseif max_opponent_score_seen == 4 then
            -- Opponent held Advantage → opponent won.
            player_won_game = false
        elseif last_player_score >= 3 and last_opponent_score < 3 then
            -- Player was at 40+; opponent was not → player won normally.
            player_won_game = true
        elseif last_opponent_score >= 3 and last_player_score < 3 then
            -- Opponent was at 40+; player was not → opponent won normally.
            player_won_game = false
        elseif last_player_score > last_opponent_score then
            player_won_game = true
        else
            player_won_game = false
        end

        if player_won_game then
            actual_player_games = actual_player_games + 1
            reward = reward + REWARD_GAME_WIN
            if actual_player_games >= MATCH_GAMES then
                reward = reward + REWARD_MATCH_WIN
            end
        else
            actual_opponent_games = actual_opponent_games + 1
            reward = reward + REWARD_GAME_LOSS
            if actual_opponent_games >= MATCH_GAMES then
                reward = reward + REWARD_MATCH_LOSS
            end
        end

        -- Reset sticky peaks for the next game.
        max_player_score_seen   = 0
        max_opponent_score_seen = 0
    else
        -- Within-game point detection via score changes.
        if player_score > last_player_score then
            -- Player scored a point (normal progression: 0→1→2→3).
            reward = reward + REWARD_POINT_WIN
        elseif opponent_score > last_opponent_score then
            -- Opponent scored a point.
            reward = reward + REWARD_POINT_LOSS
        elseif last_player_score == 4 and player_score == 3 then
            -- Player lost Advantage → back to Deuce; opponent scored.
            reward = reward + REWARD_POINT_LOSS
        elseif last_opponent_score == 4 and opponent_score == 3 then
            -- Opponent lost Advantage → back to Deuce; player scored.
            reward = reward + REWARD_POINT_WIN
        end
    end

    -- Episode ends when either side has won MATCH_GAMES games.
    local done = actual_player_games >= MATCH_GAMES or actual_opponent_games >= MATCH_GAMES

    last_player_score   = player_score
    last_opponent_score = opponent_score
    last_point_end      = point_end
    return reward, done,
           player_score,   actual_player_games,
           opponent_score, actual_opponent_games,
           point_end
end

local function send_observation()
    local frame = capture_frame_gray()
    if frame == nil then return false end
    local native_frame = ""
    if CAPTURE_NATIVE_FRAMES then
        native_frame = capture_frame_rgb()
        if native_frame == nil then return false end
    end
    local reward, done, player_score, player_games, opponent_score, opponent_games, point_end = compute_reward()
    if pending_token == nil then
        report_error("Observation requested without a pending host token")
        emu.stop(35)
        return false
    end
    local token = pending_token
    local header
    if RECORD_MODE then
        header = string.format(
            "O %s %.6f %d %d %d %d %d %d %d %d %d\n",
            token, reward, done and 1 or 0, player_score, player_games, opponent_score, opponent_games,
            point_end, interval_action, interval_a_frames, interval_b_frames
        )
    else
        header = string.format(
            "O %s %.6f %d %d %d %d %d %d\n",
            token, reward, done and 1 or 0, player_score, player_games, opponent_score, opponent_games, point_end
        )
    end
    if RECORD_MODE then
        interval_action = 0
        interval_a_frames = 0
        interval_b_frames = 0
    end
    local suffix = token
    local obs_path = OBS_PREFIX .. suffix .. ".bin"
    local obs_tmp = OBS_PREFIX .. suffix .. ".tmp"
    local ready_path = OBS_READY_PREFIX .. suffix .. ".ready"
    local ready_tmp = OBS_READY_PREFIX .. suffix .. ".tmp"
    os.remove(ready_path)
    os.remove(obs_path)
    if not atomic_write(obs_path, obs_tmp, header .. frame .. native_frame) then
        report_error("Failed to write observation mailbox: " .. obs_path)
        stopped = true
        pcall(function() emu.stop(31) end)
        return false
    end
    if not atomic_write(ready_path, ready_tmp, "READY\n") then
        report_error("Failed to publish observation readiness: " .. ready_path)
        stopped = true
        pcall(function() emu.stop(31) end)
        return false
    end
    return true
end

local function button_action_index(input)
    local movement = 0
    local left = input.left == true
    local right = input.right == true
    local up = input.up == true
    local down = input.down == true
    if left and up then movement = 5
    elseif right and up then movement = 6
    elseif left and down then movement = 7
    elseif right and down then movement = 8
    elseif left then movement = 1
    elseif right then movement = 2
    elseif up then movement = 3
    elseif down then movement = 4
    end
    local shot = (input.a == true) and 1 or ((input.b == true) and 2 or 0)
    return movement * 3 + shot
end

local input_callback = emu.addEventCallback(function()
    if RECORD_MODE and boot_complete then
        local input = emu.getInput(0)
        human_action = button_action_index(input)
        -- Record the last non-neutral controller command observed during the
        -- FRAME_SKIP interval. This preserves short A/B taps that may have
        -- been released before the observation boundary.
        if human_action ~= 0 then
            interval_action = human_action
        end
    else
        -- During automatic boot, even record mode must drive the menu. Once the
        -- match is ready, record mode stops overriding player-1 input and the
        -- human remains in control.
        emu.setInput(0, current_input())
    end
end, emu.eventType.inputPolled)

local function reset_tracking()
    initialized = false
    frame_counter = 0
    step_pending = false
    pending_token = nil
    current_action = 0
    current_pulse_frames = -1
    current_pulse_remaining = 0
    actual_player_games = 0
    actual_opponent_games = 0
    max_player_score_seen = 0
    max_opponent_score_seen = 0
    frames_since_progress = 0
    last_frame_player_score = -1
    last_frame_opponent_score = -1
    stall_rescue_count = 0
    for k, _ in pairs(held_buttons) do held_buttons[k] = 0 end
end

local function write_ready()
    if ready_written then return end
    local ok = write_text(READY_PATH, string.format("READY %d %d %d\n", WIDTH, HEIGHT, FRAME_SKIP))
    if not ok then
        report_error("Unable to initialize IPC mailbox; enable Mesen Lua I/O/OS access")
        emu.stop(32)
        return
    end
    ready_written = true
end

local function read_binary(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local data = f:read("*a")
    f:close()
    return data
end

local function write_binary(path, data)
    local f, err = io.open(path, "wb")
    if not f then return false, tostring(err) end
    f:write(data)
    f:close()
    return true
end

-- Mesen 2 documents both createSavestate() and loadSavestate() as CPU-exec-only.
-- We register a one-shot callback only when a state request is pending, so normal
-- operation does not pay the cost of a callback on every CPU instruction.
local SAVE_FN = emu.saveSavestate or emu.createSavestate
local function create_state()
    if SAVE_FN == nil then
        error("this Mesen build has neither emu.saveSavestate nor emu.createSavestate")
    end
    return SAVE_FN()
end

local function perform_state_op_in_exec()
    if stopped then return end
    if snapshot_pending then
        snapshot_pending = false
        local ok, err = pcall(function() match_savestate = create_state() end)
        if not ok or match_savestate == nil then
            report_error("Failed to create match-start savestate (R/LOAD_FROM disabled): " .. tostring(err))
        end
        write_ready()
    end

    local op = pending_op
    if not op then return end
    pending_op = nil
    state_op_wait = 0
    if op.kind == "save" then
        local ok, err = pcall(function() saved_state = create_state() end)
        if ok and saved_state ~= nil then
            write_text(IPC_DIR .. "/state_saved.txt", "SAVED\n")
        else
            report_error("Failed to create savestate: " .. tostring(err))
        end
    elseif op.kind == "save_to" then
        local ok, data = pcall(create_state)
        if not ok or not data then
            report_error("Failed to create savestate for SAVE_TO: " .. tostring(data))
        else
            local wrote, werr = write_binary(op.path, data)
            if wrote then
                write_text(IPC_DIR .. "/state_saved.txt", "SAVED\n")
            else
                report_error("Could not write savestate file " .. tostring(op.path) .. ": " .. tostring(werr))
            end
        end
    else
        local data
        if op.kind == "reset" then data = match_savestate
        elseif op.kind == "load" then data = saved_state
        else data = read_binary(op.path) end
        local ok, loaded = false, false
        if data ~= nil then
            ok, loaded = pcall(function() return emu.loadSavestate(data) end)
        else
            report_error("No savestate data for " .. op.kind .. " " .. tostring(op.path))
        end
        if ok and loaded ~= false then
            reset_tracking()
            if op.kind == "load_from" then
                -- A LOAD_FROM acknowledgement denotes the state plus exactly N
                -- subsequent no-op frame boundaries, then a stable lockstep barrier.
                noop_remaining = math.max(0, math.floor(op.noop or 0))
                noop_ack_pending = true
            else
                write_text(IPC_DIR .. "/state_loaded.txt", "LOADED\n")
            end
        elseif data ~= nil then
            report_error("Failed to load savestate (" .. op.kind .. "): " .. tostring(loaded))
        end
    end
end

local function main_cpu_program_counter()
    -- Mesen 2 exposes the current NES CPU PC through getState().cpu.pc. Keep
    -- compatibility fallbacks for builds whose getCpuState() table is flatter.
    local ok, state = pcall(function() return emu.getState() end)
    if ok and type(state) == "table" then
        local cpu = state.cpu
        if type(cpu) == "table" then
            local pc = tonumber(cpu.pc or cpu.PC or cpu.programCounter)
            if pc and pc >= 0 and pc <= 0xFFFF then return math.floor(pc) end
        end
    end
    ok, state = pcall(function() return emu.getCpuState() end)
    if ok then
        if type(state) == "number" then
            if state >= 0 and state <= 0xFFFF then return math.floor(state) end
        elseif type(state) == "table" then
            local pc = tonumber(state.pc or state.PC or state.programCounter)
            if pc and pc >= 0 and pc <= 0xFFFF then return math.floor(pc) end
        end
    end
    return nil
end

local function disarm_state_op_callback()
    local id = state_op_callback_id
    if id == nil then
        state_op_callback_fired = false
        state_op_callback_address = nil
        return true
    end

    -- Do not remove a memory callback from inside that same memory callback:
    -- Mesen iterates its callback list during dispatch. Removal is done by the
    -- next startFrame/endFrame event, outside CPU memory-callback dispatch.
    local ok, err = pcall(function()
        emu.removeMemoryCallback(id, EXEC_CALLBACK_TYPE, state_op_callback_address, state_op_callback_address)
    end)
    if not ok then
        report_error("Could not disarm completed savestate exec callback: " .. tostring(err))
        stopped = true
        pcall(function() emu.stop(37) end)
        return false
    end
    state_op_callback_id = nil
    state_op_callback_fired = false
    state_op_callback_address = nil
    return true
end

local function arm_state_op_callback()
    if stopped then return end
    -- Remove the previous instruction callback only after it has fired and
    -- returned to a frame event. This avoids mutating Mesen's callback array
    -- while it is being iterated by the CPU.
    if state_op_callback_id ~= nil then
        if state_op_callback_fired then
            if not disarm_state_op_callback() then return end
        else
            return
        end
    end
    if not snapshot_pending and pending_op == nil then return end
    if type(emu.addMemoryCallback) ~= "function" or EXEC_CALLBACK_TYPE == nil then
        local msg = "Savestate operations require emu.addMemoryCallback and an exec callback type"
        report_error(msg)
        stopped = true
        pcall(function() emu.stop(37) end)
        return
    end

    local pc = main_cpu_program_counter()
    if pc == nil then
        local msg = "Could not read main CPU program counter to schedule savestate operation"
        report_error(msg)
        stopped = true
        pcall(function() emu.stop(37) end)
        return
    end

    local fired = false
    local function one_shot_exec_callback()
        if fired then return end
        fired = true
        state_op_callback_fired = true
        local ok, err = pcall(perform_state_op_in_exec)
        if not ok then
            report_error("Savestate operation callback failed: " .. tostring(err))
            stopped = true
            pcall(function() emu.stop(37) end)
        end
    end

    -- Register only at the next CPU instruction's address. A 0..0xFFFF exec
    -- callback would be invoked for every instruction in the frame after the
    -- first operation, causing severe slowdown until endFrame removes it.
    local ok, id = pcall(function()
        return emu.addMemoryCallback(one_shot_exec_callback, EXEC_CALLBACK_TYPE, pc, pc)
    end)
    if not ok or id == nil then
        report_error("Could not register one-shot exec callback for savestate operation: " .. tostring(id))
        stopped = true
        pcall(function() emu.stop(37) end)
        return
    end
    state_op_callback_id = id
    state_op_callback_address = pc
end

-- startFrame is only the safe time to arm a deferred CPU-exec callback; the
-- savestate API itself runs inside that callback, as required by Mesen 2.
pcall(function()
    emu.addEventCallback(arm_state_op_callback, emu.eventType.startFrame)
end)

-- Write the API names this Mesen build exposes, so a mismatch is diagnosable from
-- api_info.txt instead of guessed at.
local function write_api_info()
    local lines, names = {}, {}
    for k, _ in pairs(emu) do names[#names + 1] = tostring(k) end
    table.sort(names)
    lines[#lines + 1] = "emu: " .. table.concat(names, ", ")
    for _, enum_name in ipairs({"eventType", "memType", "callbackType", "cpuType"}) do
        local e = emu[enum_name]
        if type(e) == "table" then
            local m = {}
            for k, v in pairs(e) do m[#m + 1] = tostring(k) .. "=" .. tostring(v) end
            table.sort(m)
            lines[#lines + 1] = "emu." .. enum_name .. ": " .. table.concat(m, ", ")
        else
            lines[#lines + 1] = "emu." .. enum_name .. ": (absent)"
        end
    end
    lines[#lines + 1] = "savestate function: " .. (emu.saveSavestate and "saveSavestate" or (emu.createSavestate and "createSavestate" or "NONE"))
    lines[#lines + 1] = "lockstep=" .. tostring(LOCKSTEP) .. " slice=" .. tostring(LOCKSTEP_SLICE_S) .. "s"
    lines[#lines + 1] = "strict_lockstep=" .. tostring(STRICT_LOCKSTEP) .. " wait=" .. tostring(STRICT_LOCKSTEP_WAIT_S) .. "s"
    write_text(IPC_DIR .. "/api_info.txt", table.concat(lines, "\n") .. "\n")
end

local function has_held_button()
    for _, frames in pairs(held_buttons) do
        if frames > 0 then return true end
    end
    return false
end

-- True while the emulator must keep running frames even though no step is pending.
local function needs_frames()
    return pending_op ~= nil or snapshot_pending or noop_remaining > 0 or noop_ack_pending
end

local function lockstep_wait()
    if not LOCKSTEP or needs_frames() then return true end
    local t0 = os.clock()
    local spins = 0
    while not stopped do
        local line = read_command()
        if line then
            apply_command(line)
            if stopped then return false end
            if step_pending or has_held_button() or needs_frames() then return true end
            t0 = os.clock()  -- e.g. PING: keep waiting for a real step
        else
            spins = spins + 1
            if spins >= 32 then
                spins = 0
                -- Lupa-based tests embed Lua in the same process as the Python
                -- host thread. Some Lupa builds keep the GIL while executing Lua,
                -- so a pure Lua busy loop can prevent the host thread from writing
                -- command.in. Production Mesen runs in a separate process and does
                -- not define this test-only hook.
                local test_yield = rawget(_G, "__MESEN_RL_TEST_YIELD")
                if type(test_yield) == "function" then pcall(test_yield) end
                local elapsed = os.clock() - t0
                if STRICT_LOCKSTEP then
                    -- Scenario replay must be reproducible. A timeout-driven frame
                    -- would silently alter the next action's state, so stop loudly.
                    if elapsed >= STRICT_LOCKSTEP_WAIT_S then
                        local msg = string.format(
                            "strict lockstep host-command timeout after %.3fs; emulator stopped to preserve replay determinism",
                            elapsed
                        )
                        report_error(msg)
                        stopped = true
                        pcall(function() emu.stop(38) end)
                        return false
                    end
                elseif elapsed >= LOCKSTEP_SLICE_S then
                    if os.time() - last_host_activity > LOCKSTEP_TIMEOUT_S then
                        stopped = true
                        report_error("host silent for " .. tostring(LOCKSTEP_TIMEOUT_S) .. "s; stopping emulator")
                        emu.stop(36)
                        return false
                    end
                    return true  -- soft-lockstep yields one frame to avoid timing out learner updates
                end
            end
        end
    end
    return false
end

local end_callback = emu.addEventCallback(function()
    if state_op_callback_fired then
        if not disarm_state_op_callback() then return end
    end
    if stopped then return end

    local ok, err = pcall(function()
        if not boot_complete then
            return
        end

        -- Update sticky peak point-scores every frame so compute_reward() can
        -- determine game winners even when the Advantage frame (score==4) falls
        -- between two FRAME_SKIP sample boundaries.
        if initialized then
            local ps = read_u8(RAM_PLAYER_SCORE)
            local opp_s = read_u8(RAM_OPPONENT_SCORE)
            max_player_score_seen   = math.max(max_player_score_seen,   ps)
            max_opponent_score_seen = math.max(max_opponent_score_seen, opp_s)

            -- Stall watchdog (see STALL_FRAMES_BEFORE_START above). Alternates
            -- between Start and A: we don't know for certain whether a stall
            -- is a between-point/game overlay (needs Start to dismiss) or a
            -- serve/rally waiting on a swing input (needs A), so try both in
            -- turn rather than assuming.
            if ps == last_frame_player_score and opp_s == last_frame_opponent_score then
                frames_since_progress = frames_since_progress + 1
                if frames_since_progress >= STALL_FRAMES_BEFORE_START then
                    local rescue_button = (stall_rescue_count % 2 == 0) and "start" or "a"
                    held_buttons[rescue_button] = math.max(held_buttons[rescue_button] or 0, STALL_START_HOLD_FRAMES)
                    stall_rescue_count = stall_rescue_count + 1
                    frames_since_progress = 0
                end
            else
                frames_since_progress = 0
            end
            last_frame_player_score = ps
            last_frame_opponent_score = opp_s
        end

        -- Poll commands every emulator frame. This keeps reset/action latency
        -- deterministic under --testrunner; the expensive operation (frame
        -- serialization) remains limited to FRAME_SKIP frames below.
        if LOCKSTEP and not step_pending and not has_held_button() and not needs_frames() then
            -- Idle (after boot / between steps): don't let the game run ahead of the host.
            local wait_ok = lockstep_wait()
            if not wait_ok or stopped then return end
        else
            poll_command()
        end

        -- Safety nets: never leave the host waiting forever for READY or a savestate.
        if not ready_written then
            boot_complete_frames = boot_complete_frames + 1
            if boot_complete_frames > 180 then
                report_error("savestate snapshot never completed; R/LOAD_FROM unavailable")
                snapshot_pending = false
                write_ready()
            end
        end
        if pending_op ~= nil then
            state_op_wait = state_op_wait + 1
            if state_op_wait > 180 then
                report_error("savestate request not serviced after 180 frames; dropped")
                pending_op = nil
                state_op_wait = 0
            end
        else
            state_op_wait = 0
        end

        -- No-op frames after LOAD_FROM run free (no input, no observation).
        if noop_remaining > 0 then
            current_action = 0
            noop_remaining = noop_remaining - 1
        end

        -- Acknowledge LOAD_FROM only at a stable end-of-frame boundary, including
        -- LOAD_FROM 0. Without this barrier, the host may wake after an ack while
        -- lockstep timeout slices let arbitrary idle frames elapse. Commands latched
        -- here take effect at the next startFrame, and this boundary is never counted
        -- as part of a newly received action or HOLD duration.
        if noop_remaining == 0 and noop_ack_pending then
            noop_ack_pending = false
            write_text(IPC_DIR .. "/state_loaded.txt", "LOADED\n")
            if LOCKSTEP then
                local wait_ok = lockstep_wait()
                if not wait_ok or stopped then return end
            end
            return
        end

        -- Count down precise A/B pulses per emulated NES frame. Movement remains
        -- held for the complete observation interval.
        if current_pulse_frames > 0 and current_pulse_remaining > 0 then
            current_pulse_remaining = current_pulse_remaining - 1
        end

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
            if frame_counter < FRAME_SKIP then frame_counter = frame_counter + 1 end
            if frame_counter >= FRAME_SKIP then
                local sent = send_observation()
                if sent then
                    step_pending = false
                    pending_token = nil
                    frame_counter = 0
                    screen_capture_wait_frames = 0
                    last_host_activity = os.time()
                    local wait_ok = lockstep_wait()
                    if not wait_ok or stopped then return end
                else
                    -- Startup / savestate transitions can temporarily expose an
                    -- empty framebuffer. Preserve the token and retry on the next
                    -- rendered frame instead of terminating Mesen on a nil pixel.
                    screen_capture_wait_frames = screen_capture_wait_frames + 1
                    if screen_capture_wait_frames > 180 then
                        report_error("Screen buffer remained unavailable or incomplete for 180 frames while producing observation token " .. tostring(pending_token))
                        stopped = true
                        pcall(function() emu.stop(34) end)
                        return
                    end
                end
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
    pcall(write_api_info)
    if not AUTO_BOOT then
        local ok = write_text(READY_PATH, string.format("READY %d %d %d\n", WIDTH, HEIGHT, FRAME_SKIP))
        if not ok then
            report_error("Unable to initialize IPC mailbox; enable Mesen Lua I/O/OS access")
            emu.stop(32)
        else
            ready_written = true
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
