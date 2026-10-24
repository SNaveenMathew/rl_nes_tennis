# Training the hierarchical NES Tennis agent (Windows)

This guide describes the staged hierarchy introduced in this repository. Start from fresh hierarchy checkpoints; old flat-PPO or seven-skill checkpoints are not compatible and should not be used with this hierarchy.

## 0. Mesen version and paths

The current `configs/scenario.json` is set to the paths shown in the working Windows setup: `C:\Games\NES\Mesen2.2.1\Mesen.exe` and `C:\Users\dbznf\Documents\NES\Tennis (JU).nes`. If you install a different Mesen 2.2.1 or MesenCE build, update those paths rather than pointing at the ZIP itself. The bridge relies on the documented `emu.callbackType.exec`, `emu.getState().cpu.pc`, and ARGB `emu.getScreenBuffer()` APIs. Run `scripts/check_mesen.ps1` against your exact executable/ROM because a successful mock test does not establish live-emulator compatibility.

For MesenCE, the normal Windows download is a ZIP containing the `Mesen.exe` launcher and supporting files. Verify any downloaded build through its official release source. Compatibility must be checked on the exact executable installed on the laptop.

## 1. Environment and emulator smoke check

From PowerShell:

```powershell
cd C:\Coding\tennis_rl_production
# Activate the Python environment used for this project, then:
pip install -r requirements.txt
python -m pytest -q
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical --workers 1 --smoke-test
```

If your local CLI does not support `--smoke-test`, use `python -m tennis_rl.train --help` and run the repository's bridge smoke-test script in `scripts/`. A successful smoke check must show a valid observation of the expected shape, at least one request/response exchange, and no bridge startup error. For the first run, use one worker; only increase workers after checking for timeouts.

## 2. Record demonstrations and scenario states

The best initial signal comes from human traces with exact task labels. Record complete, separate matches so whole matches can be held out from training. The recording command opens Mesen in interactive mode:

```powershell
python -m tennis_rl.record --config configs/scenario.json --output demonstrations --harvest-dir scenario_states --harvest-every 8 --serve serve-first --scenario-tag serve_and_rally_level1
python -m tennis_rl.record --config configs/scenario.json --output demonstrations --harvest-dir scenario_states --harvest-every 8 --serve receive-first --scenario-tag receive_and_rally_level1
```

Play to the end of the requested recorded game; pressing Ctrl+C before a game ends intentionally discards incomplete buffered points. The receiver recording skips game 1, then records game 2 when the CPU serves. For broader coverage repeat at several game levels and record separate matches. Scenario savestates are stored in `scenario_states/manifest.jsonl` and `scenario_states/states/`.

Start with approximately 20–50 complete points across both service roles to validate the recorder, bridge, and annotation workflow; that is not sufficient by itself to train all 13 leaves. As an initial diagnostic target, aim for at least 100 explicitly labeled observation windows per leaf across several matches, and substantially more for rare or visually ambiguous events such as high lobs. Treat that only as a starting floor: increase the count until held-out task accuracy and real-point performance stabilize.

## 3. Audit and extract segments

```powershell
python -m tennis_rl.dataset stats --demo-root demonstrations
python -m tennis_rl.dataset split --demo-root demonstrations --holdout-fraction 0.2
python -m tennis_rl.dataset extract --demo-root demonstrations --segment-length 16 --stride 8
python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --make-sheets
python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --interactive --state-labels
python -m tennis_rl.dataset stats --demo-root demonstrations
```

`stats` reports point files/outcomes, matches, scenario tags, per-task annotation counts, known button-pulse labels, and missing pulse coverage. `extract` writes only segment metadata (the frame/action arrays stay in original files), and `split` assigns whole matches to train or holdout. Avoid one-match train/holdout splits; gather multiple separate matches so both sets have examples of the important tasks.

In the annotation prompt, use half-open intervals `start,end,task`, where `end` is exclusive. Example leaf labels include `serve_timing`, `receive_position`, `intercept_track`, `hold_cover`, `retreat_lob`, `stroke_selector`, and `stroke_timing`. Legacy labels such as `serve`, `receive`, `baseline`, and `wait` are only broad phase supervision; they never fabricate leaf action targets. Add optional `manager_duration` (1, 2, 4, or 8 observation steps) only when you can identify a deliberate skill execution duration. Leave it unknown otherwise.

For state supervision, annotate selected frames with normalized `[x,y]` locations for the ball and player, signed ball velocity, approximate time-to-contact, and only those event labels that you can actually verify. Blank means unknown—not no. The model will mask missing labels. Do not infer ball coordinates from the existing motion-centroid heuristic.

## 4. Train shared foundation and leaf specialists

### 4.1 Foundation

The foundation jointly warms up the shared visual encoder, recurrent belief state, all leaf heads, and the manager using only explicit labels that exist:

```powershell
python -m tennis_rl.staged_train --config configs/scenario.json --stage foundation --split train --updates 1000 --batch-size 64 --eval-every 50 --output checkpoints_hierarchical_native256x240_frame1/stages/foundation.pt --device cuda
```

For a quick pipeline smoke test use `--updates 3 --batch-size 8 --eval-every 1`; do not use those tiny settings for a real model.

If no leaf/phase labels exist, this stage exits rather than synthesizing target actions. If no state annotations exist, the perception part is simply masked; to train `--stage perception`, `state_annotations.jsonl` must contain valid labels.

### 4.2 Optional state-perception training

If `state_annotations.jsonl` contains enough verified coordinates/events, improve the shared encoder before training specialists. Every leaf must then be trained from this same updated checkpoint so its frozen encoder weights match at assembly time. If you have no trustworthy ball/player/contact labels yet, skip this stage rather than inventing labels; use the foundation checkpoint as the shared starting point and prioritize collecting/annotating examples.

```powershell
python -m tennis_rl.staged_train --config configs/scenario.json --stage perception --init-from checkpoints_hierarchical_native256x240_frame1/stages/foundation.pt --split train --updates 500 --batch-size 64 --eval-every 25 --output checkpoints_hierarchical_native256x240_frame1/stages/perception.pt --device cuda
```

### 4.3 Independent leaf training

Each leaf starts from the *same shared checkpoint*: `perception.pt` if you trained the optional perception stage, otherwise `foundation.pt`. Its encoder and all other experts are frozen, which makes separately trained experts compatible for assembly:

Use one shared checkpoint path consistently for **all** leaves and assembly. Default (no state-perception stage):

```powershell
$Shared = "checkpoints_hierarchical_native256x240_frame1/stages/foundation.pt"
python -m tennis_rl.staged_train --config configs/scenario.json --stage leaf --skill serve_timing --init-from $Shared --split train --updates 1000 --batch-size 64 --eval-every 50 --output checkpoints_hierarchical_native256x240_frame1/stages/leaf_serve_timing.pt --device cuda
python -m tennis_rl.staged_train --config configs/scenario.json --stage leaf --skill receive_position --init-from $Shared --split train --updates 1000 --batch-size 64 --eval-every 50 --output checkpoints_hierarchical_native256x240_frame1/stages/leaf_receive_position.pt --device cuda
```

If and only if you trained the optional perception stage, set `$Shared = "checkpoints_hierarchical_native256x240_frame1/stages/perception.pt"` instead. Every leaf must start from that same exact base. Repeat for each task with adequate exact action annotations. The canonical task list is in `tennis_rl/hierarchy_schema.py`. You may train a subset during development, but a task left at the foundation weights is **not** a validated specialist. To run all thirteen tasks as a script (it deliberately stops at the first missing task annotation), use:

```powershell
.\scripts\train_leaf_experts.ps1 -Config configs/scenario.json -DemoRoot demonstrations -Foundation $Shared
```

### 4.4 Assemble the specialists

```powershell
python -m tennis_rl.assemble_hierarchy --config configs/scenario.json --base $Shared `
  --leaf serve_timing=checkpoints_hierarchical_native256x240_frame1/stages/leaf_serve_timing.pt `
  --leaf receive_position=checkpoints_hierarchical_native256x240_frame1/stages/leaf_receive_position.pt `
  --output checkpoints_hierarchical_native256x240_frame1/stages/leaves_assembled.pt
```

Add each trained leaf using another `--leaf TASK=PATH` argument. The assembler verifies the task list, button pulse schema, expert parameter schema, and that shared encoder weights are identical; it refuses to silently merge incompatible checkpoints.

### 4.5 Train the routing gates

```powershell
python -m tennis_rl.staged_train --config configs/scenario.json --stage gates --init-from checkpoints_hierarchical_native256x240_frame1/stages/leaves_assembled.pt --split train --updates 1000 --batch-size 64 --eval-every 50 --output checkpoints_hierarchical_native256x240_frame1/stages/gated.pt --device cuda
```

Gate supervision must have broad phase labels (serve/receive/rally/transition) and exact leaf labels for sub-gate selection. It only learns explicit annotations. Skill duration is supervised only when `manager_duration` was explicitly labeled; otherwise duration is learned from online RL.

## 5. Evaluate every stage independently

Leaf-level evaluations force the selected expert. The manager evaluation uses `--skill auto`. Each valid manifest state is restored and played for multiple full point attempts; logs include point outcomes, rewards, steps, faults, and bridge restarts.

```powershell
python -m tennis_rl.evaluate_scenarios --config configs/scenario.json --checkpoint checkpoints_hierarchical_native256x240_frame1/stages/leaf_serve_timing.pt --skill serve_timing --manifest scenario_states/manifest.jsonl --split holdout --repeats 5 --device cuda --out runs_hierarchical_native256x240_frame1/eval_serve.jsonl
python -m tennis_rl.evaluate_scenarios --config configs/scenario.json --checkpoint checkpoints_hierarchical_native256x240_frame1/stages/gated.pt --skill auto --manifest scenario_states/manifest.jsonl --split holdout --repeats 5 --device cuda --out runs_hierarchical_native256x240_frame1/eval_auto.jsonl
```

For diagnostic skills, evaluator filters to compatible scenario state types. If no compatible holdout state exists, gather more data rather than switching silently to training states. Compare checkpoints with the same manifest, split, repeat count, seed and step horizon. Do not score truncated runs as wins or losses.

## 6. Online hierarchical PPO fine-tuning

Only after the leaf skills and gates show useful held-out behavior, fine-tune the assembled hierarchy using actual point-level outcomes:

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical --init-from checkpoints_hierarchical_native256x240_frame1/stages/gated.pt --workers 4 --max-updates 100
```

`--init-from` loads weights only, starting a fresh optimizer. `--resume` is for continuing a previous online run with its optimizer/counters; it is not required for this staged-from-scratch workflow. Increase workers only after bridge stability tests. Use fixed held-out scenarios regularly; checkpoint every few updates early in development.

## Recommended task coverage

| Phase | Leaf tasks | Important contrast pairs |
|---|---|---|
| Serve | `serve_timing`, `serve_placement_power`, `serve_recovery` | initiate now/wait; valid/fault; good/poor placement; recovery/no recovery |
| Receive | `receive_position`, `return_stroke_timing` | fast/slow serve; left/right; short/deep; lob/normal return where legal |
| Rally | `intercept_track`, `hold_cover`, `recover_reposition`, `approach_net`, `retreat_lob`, `stroke_selector`, `stroke_timing` | toward/away; reachable/missed; move/wait; baseline/net; lob/no lob; early/correct/late stroke |
| Transition | `transition_wait_reposition` | CPU turn/agent turn; dead time/serve-ready; wait/reposition |

Collect both positive and negative examples, across levels, service roles, point scores, player court sides, and varied trajectories. Avoid overfitting by splitting at the match level and reporting per-task sample counts and held-out accuracy alongside full-point win rate.

## What is and is not automated

- Segment extraction, match-safe splits, annotation validation, masked losses, checkpoint assembly, and repeatable evaluator runs are automated.
- The repository intentionally does **not** hallucinate ball/contact labels from pixels or infer that an action was correct just because the point was won.
- Some scenarios and event labels require a human to play/annotate real examples. The current tool does not automatically recognize all ROM sprite/contact events.
- Synthetic smoke training validates that the software can optimize a known toy target. Only evaluation against the actual ROM can establish tennis-playing improvement.
