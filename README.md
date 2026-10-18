# Tennis RL

A Windows-first reinforcement-learning framework for the 1984 NES Tennis ROM running in Mesen/MesenCE. The default training path is a **learned hierarchical PPO policy**: a global manager chooses a skill and a learned duration, and an independent specialist policy emits controller actions. Flat recurrent PPO and DQfD remain available as baselines.

> **Validation boundary:** this repository's sandbox smoke tests use synthetic observations and a mock worker. They validate code paths, optimizer updates, checkpoint round-trips, and supervised-loss behavior, but they do **not** establish that the policy can win real NES Tennis matches. Real-game validation requires the local Mesen executable, the ROM, and held-out emulator matches.

## Architecture

- **Mesen bridge and IPC** (`mesen/bridge.lua`, `tennis_rl/mesen_client.py`): frame capture, controller transport, score signals, and lock-step file-mailbox IPC.
- **Vector environment** (`tennis_rl/env.py`): batched workers, frame stacks, point/scenario episodes, explicit termination vs. truncation, and role-aware serve-timeout semantics.
- **Hierarchical policy** (`tennis_rl/hierarchical.py`, `tennis_rl/hierarchical_train.py`): learned manager, semi-Markov skill/duration decisions, separate skill-specific actor-critic heads, and task losses only from explicit annotation files.
- **Visual/action model** (`tennis_rl/model.py`): CNN visual encoder and a joint 27-way controller-action distribution, allowing movement and shot inputs to interact.
- **Training data** (`tennis_rl/task_data.py`, `tennis_rl/record.py`, `tennis_rl/scenarios.py`): human demonstration recording, optional explicit task/state annotations, and scenario/state sampling. No inferred pixel centroid is treated as a ground-truth ball label.
- **Baseline algorithms** (`tennis_rl/ppo.py`, `tennis_rl/dqfd.py`): flat recurrent PPO with terminal-observation-aware truncated GAE, and an optional DQfD path retained for comparison.
- **Evaluation** (`tennis_rl/evaluate.py`, `tennis_rl/watch.py`): detects hierarchy checkpoints and routes inference through the learned manager/controller.

## Quick start

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Use a Python version supported by your installed PyTorch build. `lupa` is required by the bridge-source test harness; the NES ROM and Mesen executable are separate local assets and are not included in the ZIP.

### 2. Configure Mesen and the ROM

Edit `configs/scenario.json` to point `mesen.executable`, `mesen.rom`, and `mesen.lua_script` to files on your machine. The bundled `configs/base.json.example` is a template, not a ready-to-run local config. Reduce `mesen.workers` if your machine cannot sustain the configured concurrency.

### 3. Run sandbox-level checks (no ROM needed)

```powershell
python scripts/sandbox_train_smoke.py
python -m compileall -q tennis_rl tests
pytest -q
```

The smoke script uses a deterministic mock environment, performs multiple hierarchical PPO updates, saves/loads a checkpoint, and checks that behavior-cloning loss falls on a simple synthetic labeling task. It is a software sanity check, not evidence of game-playing improvement.

### 4. Train the hierarchical model against Mesen

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical --max-updates 3
```

`--max-updates 3` is a short **real-emulator** smoke run; it still needs the configured ROM and Mesen executable. For longer training, omit `--max-updates` or set a larger value. Checkpoint and metrics paths are controlled by the `hierarchical` configuration section (`checkpoints_hierarchical/`, `runs_hierarchical/` by default). A resumed hierarchical checkpoint should use the hierarchical algorithm:

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical --resume checkpoints_hierarchical/hierarchical_latest.pt
```

### 5. Compare against the flat baselines

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo ppo --max-updates 3
python -m tennis_rl.train --config configs/scenario.json --algo dqfd --max-updates 3
```

Use equal scenario distributions and evaluation budgets when comparing models; do not compare training return from different samplers as though it were a held-out win rate.

### 6. Evaluate a checkpoint

```powershell
python -m tennis_rl.evaluate --config configs/scenario.json --checkpoint checkpoints_hierarchical/hierarchical_latest.pt --episodes 20
```

Use only fixed held-out matches for reported win rates. Also report results by player/server role, serve and return success, completed points, match wins, and mean reward. Preserve the initial flat-PPO checkpoint for a baseline if one exists locally.

### Other commands

```powershell
python -m tennis_rl.record --config configs/scenario.json --level 1 --output demonstrations --serve serve-first
python -m tennis_rl.benchmark_env --config configs/scenario.json --seconds 10
```

See [HIERARCHICAL_TRAINING.md](HIERARCHICAL_TRAINING.md) for the model design, explicit task annotation schema, known limitations, and validation scope.

## Project structure

```
tennis_rl/
  config.py               Typed configuration schemas
  env.py                  Batched environment and reward/termination semantics
  model.py                Flat recurrent policy and joint movement/shot actions
  ppo.py                  PPO and corrected GAE helper
  hierarchical.py         Manager and specialized actor-critic policies
  hierarchical_train.py   Semi-Markov manager + expert rollout trainer
  task_data.py            Explicit task/state annotation loader
  dqfd.py                 Optional DQfD baseline
  evaluate.py, watch.py   Flat/hierarchical checkpoint inference
mesen/                    Mesen bridge (emulator application is external)
configs/                  Scenario config and example template
tests/                    Unit, mock integration, and regression tests
scripts/                  Sandbox smoke and Windows launch scripts
```

## Human demonstration recording

The training stack can record live human NES Tennis play directly through Mesen/Lua. The recorder uses Mesen's `inputPolled` callback and `getInput()` API to capture the controller state while preserving interactive control; Mesen documents both APIs and recommends applying `setInput()` from `inputPolled` for scripted control.

Start one visible level-specific recording:

```powershell
python -m tennis_rl.record --config configs/scenario.json --level 1 --output demonstrations --serve serve-first
```

To practice receiving first, use:

```powershell
python -m tennis_rl.record --config configs/scenario.json --level 1 --output demonstrations --serve receive-first
```

`receive-first` runs Game 1 as a warm-up, starts recording when Game 2 begins, and stops after Game 2. NES Tennis documents that Player 1 serves first and players alternate service games.

The recorder saves point-level episodes under:

```text
demonstrations/
  level_1/
    winning/
      match_<id>/
        point_0001.npz
        point_0002.npz
        match.json
    losing/
      match_<id>/
        ...
  level_2/
  ...
  level_5/
```

Each point file contains stacked 84x84 observations, the 27-action controller label, reward, done flag, and metadata. The match outcome determines the `winning`/`losing` directory; a lost match is not treated as if every action inside it were wrong.

### Training with demonstrations

The hierarchy's imitation and perception losses use only **explicit annotation sidecars** (`task_annotations.jsonl` and `state_annotations.jsonl`) in `demonstrations/`; see [HIERARCHICAL_TRAINING.md](HIERARCHICAL_TRAINING.md). The repository does not ship recorded match data or annotation sidecars.

The flat recurrent PPO baseline (`--algo ppo`) retains the older demonstration warm-start and point-window replay code. That alternate path also retains heuristic segment categories, which are useful sampling metadata but are not validated ground-truth skill labels. Do not assume this older replay path is used by the hierarchical trainer.

## Scenario-based Point Training (optional)

If `scenario.enabled` is true and `scenario_states/manifest.jsonl` exists, the environment can mix state-started point/segment episodes with full-game anchor episodes. The manifest and referenced savestates must be generated from your own Mesen recordings; neither savestates nor demonstration NPZ files are bundled.

- Scenario metadata uses explicit `serve|...` and `return|...` roles where known.
- The `start`-phase stall penalty applies only to a scenario explicitly identified as a serving episode. Receiver timeouts are truncations and do not incur a serve-fault penalty.
- `mid` and `late` phase timeout rewards are small progress signals; the GAE calculation bootstraps from the final pre-reset state on truncation.
- Scenario sampling can prioritize weak/forgotten cells, while evaluation should remain fixed and held out by source match.

Run the hierarchy (default):

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical
```

Run the alternative flat PPO or DQfD baseline for a controlled comparison:

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo ppo
python -m tennis_rl.train --config configs/scenario.json --algo dqfd
```

A hierarchy policy cannot be called a validated winning model until it is evaluated against the actual game AI over held-out emulator matches. Synthetic smoke-test metrics demonstrate only that the training pipeline executes and that supervised loss can decrease on a controlled mock task.
