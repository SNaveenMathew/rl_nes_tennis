# Point-level scenario training

Instead of learning from whole games, the agent starts from a **savestate taken at the start (or middle) of a
real point** and plays only until that point is decided. A point is ~10-60 agent steps, the reward is +1/-1 and
arrives within a few hundred frames, so credit assignment is far easier than over a 4-point game.

## 1. What a sub-episode is

A sub-episode is identified by a **cell** = (role, score), plus where in the point it starts:

* **role**: `serve` (agent serves) or `return` (CPU serves). Service alternates every game, agent first.
* **score** at point start. 18 reachable states (below) x 2 roles = **36 cells**.
* **court side** follows from the score by the rules of tennis (even points played -> deuce court, odd -> ad
  court; with the 0..4 score encoding that is parity of p+o), so "CPU serves from the deuce side" is the set of
  `return` cells with side = deuce.
* **phase**: `start` (serve-ready, full point), `mid` (in a rally), `late` (<= 45 steps before the point ended).

Cells sharing (role, side, pressure) form **20 strata**, the unit the sampler balances.

| Score (agent-CPU) | Court | Pressure |
|---|---|---|
| 0-0 | deuce | opening |
| 0-15 | ad | build up |
| 0-30 | deuce | build up |
| 0-40 | ad | game point against |
| 15-0 | ad | build up |
| 15-15 | deuce | build up |
| 15-30 | ad | build up |
| 15-40 | deuce | game point against |
| 30-0 | deuce | build up |
| 30-15 | ad | build up |
| 30-30 | deuce | build up |
| 30-40 | ad | game point against |
| 40-0 | ad | game point for |
| 40-15 | deuce | game point for |
| 40-30 | ad | game point for |
| deuce | deuce | deuce |
| Ad-in | ad | adv for |
| Ad-out | ad | adv against |

Full machine-readable list: `scenario_catalog.json` (regenerate with `tennis_rl.scenarios.write_catalog`).

Overlays (not extra cells; they are tags used for analysis and sampling):
* **demo outcome**: did the human win or lose the point the state was cut from (`win` / `loss` / `none`).
* **length bucket**: short (<=20 steps), medium (<=75), long - a proxy for quick finish vs. long rally. The repo has
  no validated ball/contact decoder, so *how* a point ended (ace, double fault, unforced error) is deliberately not labelled.
* **level**: CPU level 1..5; keep level 1 until it is beaten, then add level 2+ as a separate curriculum axis.

If the ROM also switches ends after odd games, "near/far end" is another visual axis (the CNN is not flip-invariant);
check it when you look at the harvested states and, if so, harvest games 1-4 so every (server, end) pair appears.

## 2. Relation to the demonstrations (winning / losing points)

`python -m tennis_rl.scenario_index --demo-root demonstrations` replays every recorded game with the rules of
tennis and writes `scenario_index.json`: for each point its start cell, true result (sign of its reward), length bucket,
terminal window (the last 45 steps, where credit/blame lives) and a coverage matrix cell x {win, loss}.
Cells with no demonstrations are listed, so you know which situations still need recording.

**Label bug fixed on the way:** the recorder stores the *match* result in every point file, and demonstrations.py used it
as the point label, so a point won inside a lost game was imitated as "losing" and a point lost inside a won game was
imitated at double weight. `ppo.demo_outcome_source` (default `"point"`) now labels each point by its own reward;
`"match"` restores the old behaviour.

## 3. Sampling scheme (`scenario_sampler.py`)

Each new episode is drawn as follows.

1. **Anchor** (15%): an ordinary full game from the match start. Keeps the natural state distribution in the replay
   buffer and in the policy's experience, and gives an honest full-game signal.
2. **Probe** (10% of episodes, separate draw): starts from **held-out** states (split by *source match*), plays greedy,
   and is **never added to the replay buffer**. This is the generalisation / forgetting thermometer.
3. Otherwise a **scenario**:
   * stratum probability = 40% uniform + 40% rank-prioritised (low win-rate, high learning progress, stale) +
     20% natural frequency, then multiplied by (1 + 3 x share of forgotten cells) and floored at 50% of uniform, so no
     stratum ever starves and a forgetting one is re-trained quickly;
   * cell inside the stratum weighted toward low win-rate / forgotten cells;
   * **phase**: reverse curriculum, `late` first, unlock `mid` then `start` once the newest phase reaches 70% over
     >= 20 episodes; older phases stay in the mix (50% newest);
   * **state**: source match drawn uniformly first (a long match cannot dominate), 25% of the time from a point the
     human lost, then **0-30 random no-op frames** after loading (the ROM is deterministic: without this a state
     replays the same trajectory and the agent memorises it).

Failure modes this targets: *catastrophic forgetting* (floors, anchors, retention boost, permanent demo replay),
*overfitting* (match-first sampling, no-op jitter, held-out probes, `dqfd_best_probe.pt` chosen on holdout results),
*curriculum collapse* (rank-based priorities, staleness, unlock rather than replace).
Watch `scenario_report.jsonl`: if scenario win-rates climb but the probe rate or full-game evaluation does not, raise
`anchor_fraction` and harvest more matches.

## 4. Workflow

```
# 1. record points and harvest states (human plays; one game per run)
python -m tennis_rl.record --harvest-dir scenario_states --settle-steps 20 --harvest-every 15 ...
#    play serve-first AND receive-first games, at every score you can reach, several matches each (aim >= 6 matches)
# 2. look at the states (is 'start' really serve-ready? is the label right?)
python -m tennis_rl.scenario_preview --cell "return|2-1" --phase start
# 3. tag demos
python -m tennis_rl.scenario_index --demo-root demonstrations
# 4. train
python -m tennis_rl.train --config configs/scenario.json
```

## 5. Things I could not verify without your Mesen build

The bridge targets the **original Mesen 0.9.x Lua API** (the build whose `api_info.txt` lists `emu.saveSavestate`,
numeric `emu.eventType` and `emu.memCallbackType`), not Mesen2.

* Savestates: `emu.saveSavestate()` / `emu.loadSavestate()` are only allowed inside a *startFrame* event callback (or a
  CpuExec memory callback). The bridge queues `SV`, `LS`, `R`, `SAVE_TO` and `LOAD_FROM` and services them at the next
  startFrame. A state op that is not serviced within 180 frames is dropped and reported in `error.txt`.
* Script timeout: Mesen aborts a Lua callback that runs longer than 1000 ms (GUI-only setting). Lock-step therefore
  waits for the host in slices of about 0.25 s, then lets one frame run with no input and waits again. A long learner
  update costs a few idle frames, never a crash. Tune with `MESEN_RL_LOCKSTEP_SLICE`. An orphaned emulator stops
  itself after `MESEN_RL_LOCKSTEP_TIMEOUT` seconds (default 600) of host silence.
* Diagnostics: the bridge writes `api_info.txt` at startup, and a bridge timeout now prints `error.txt` and
  `api_info.txt` in the exception text (the mailbox is deleted on exit, so this is the only copy).
* `--settle-steps 20` is a guess for "serve is ready"; use the preview to tune it.
* The score encoding (0..4, 4 = advantage) and service alternation come from the bridge's comments and tennis rules.
  The index reports games whose replay does not end exactly at game-over as inconsistent.
