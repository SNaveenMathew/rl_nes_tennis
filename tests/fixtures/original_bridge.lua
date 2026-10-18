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
--   followed by WIDTH*HEIGHT raw grayscale bytes; token is host-generated.
--
-- The policy receives only video. RAM is used only for reward diagnostics/reset.

local IPC_DIR = os.getenv("MESEN_RL_IPC_DIR") or "runs/mesen_ipc/worker"
local WIDTH = tonumber(os.getenv("MESEN_RL_W") or "84")
local HEIGHT = tonumber(os.getenv("MESEN_RL_H") or "84")
local FRAME_SKIP = tonumber(os.getenv("MESEN_RL_FRAME_SKIP") or "4")
local GAME_LEVEL = tonumber(os.getenv("MESEN_RL_LEVEL") or "1")
local RECORD_MODE = os.getenv("MESEN_RL_RECORD_MODE") == "1"
local AUTO_BOOT = os.getenv("MESEN_RL_AUTO_BOOT") ~= "0"
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
local human_action = 0
local interval_action = 0

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
            actual_player_games = 0
            actual_opponent_games = 0
            max_player_score_seen = 0
            max_opponent_score_seen = 0
            frames_since_progress = 0
            last_frame_player_score = -1
            last_frame_opponent_score = -1
            stall_rescue_count = 0
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

local function capture_frame_gray()
    local buffer = emu.getScreenBuffer()
    local out = {}
    local n = 0
    for y = 0, HEIGHT - 1 do
        local sy0 = math.floor(y * SRC_H / HEIGHT)
        local sy1 = math.floor((y + 1) * SRC_H / HEIGHT)
        if sy1 <= sy0 then sy1 = sy0 + 1 end
        for x = 0, WIDTH - 1 do
            local sx0 = math.floor(x * SRC_W / WIDTH)
            local sx1 = math.floor((x + 1) * SRC_W / WIDTH)
            if sx1 <= sx0 then sx1 = sx0 + 1 end
            local sum = 0.0
            local count = 0
            for sy = sy0, sy1 - 1 do
                local row = sy * SRC_W
                for sx = sx0, sx1 - 1 do
                    local argb = buffer[row + sx]
                    if argb == nil then
                        error(string.format("screen buffer nil at x=%d y=%d idx=%d", sx, sy, row + sx))
                    end
                    local r = math.floor(argb / 0x10000) % 0x100
                    local g = math.floor(argb / 0x100) % 0x100
                    local b = argb % 0x100
                    sum = sum + (0.299 * r + 0.587 * g + 0.114 * b)
                    count = count + 1
                end
            end
            n = n + 1
            out[n] = string.char(math.floor(sum / count + 0.5))
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
    local reward, done, player_score, player_games, opponent_score, opponent_games, point_end = compute_reward()
    if pending_token == nil then
        report_error("Observation requested without a pending host token")
        emu.stop(35)
        return
    end
    local token = pending_token
    local header
    if RECORD_MODE then
        header = string.format(
            "O %s %.6f %d %d %d %d %d %d %d\n",
            token, reward, done and 1 or 0, player_score, player_games, opponent_score, opponent_games, point_end, interval_action
        )
    else
        header = string.format(
            "O %s %.6f %d %d %d %d %d %d\n",
            token, reward, done and 1 or 0, player_score, player_games, opponent_score, opponent_games, point_end
        )
    end
    if RECORD_MODE then
        interval_action = 0
    end
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
        actual_player_games = 0
        actual_opponent_games = 0
        max_player_score_seen = 0
        max_opponent_score_seen = 0
        frames_since_progress = 0
        last_frame_player_score = -1
        last_frame_opponent_score = -1
        stall_rescue_count = 0
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
