# Hierarchical NES Tennis training

## Design

The default algorithm is a learned hierarchy, not a rule-based state machine. A manager policy chooses one skill from `serve`, `receive`, `baseline`, `approach_net`, `recover`, `shot`, and `wait`, and also chooses a duration from `[1]` environment decision in the frame-perfect configuration, so the manager is reconsidered every raw NES frame. The corresponding independent expert actor-critic head selects one of 27 movement/button combinations. The manager is sampled only at learned option boundaries. There is no hard-coded inference mapping from court/ball state to a skill.

The visual encoder is shared to keep the model tractable; skill-specific policy/value heads remain separate. The manager and active expert have separate PPO loss streams. The 27-action model uses a joint action distribution with movement-shot interaction parameters rather than assuming movement and shot choice are independent.

The first release intentionally does not create separate specialist networks for every tactical subcase. `approach_net`, `recover`, `baseline`, and `wait` are distinct learned options, while specialized behavior is only promoted into a separate skill/model when there is enough labeled data and an evaluation benefit.

## What is and is not learned from labels

The manager's optional supervision uses only trusted scenario metadata at explicitly marked start states: `serve|...` can supervise `serve`; `return|...` can supervise `receive`. This metadata affects training loss only; inference does not read it. Other manager choices are learned from returns.

Task annotations are strictly opt-in and are never inferred from pixel-change centroids or simple action-frequency heuristics. This is important because a large moving-pixel region is not necessarily the tennis ball, and a movement-heavy action segment is not necessarily a net approach.

## Optional annotation files

Put these JSON Lines files in the configured `ppo.demo_root` directory (default `demonstrations/`). Paths must be relative to that directory and must remain within it.

### `task_annotations.jsonl`

Each row maps a known segment in a recorded demonstration NPZ to a task label. `end` is exclusive; `skill` must match a skill in `hierarchical.skills`.

```json
{"path":"match_0001.npz","start":120,"end":148,"skill":"receive"}
{"path":"match_0001.npz","start":148,"end":176,"skill":"baseline"}
{"path":"match_0002.npz","start":40,"end":60,"skill":"serve"}
```

These are annotation examples only. Do not label a segment as a skill unless it has actually been inspected or derived from a validated game-state event.

### `state_annotations.jsonl`

Coordinates must be normalized to `[0, 1]` using the same screen/court coordinate convention for all annotations. Unknown values may be omitted. Event labels are binary values for the event names below.

```json
{"path":"match_0001.npz","step":121,"ball_xy":[0.54,0.38],"player_xy":[0.48,0.81],"events":{"ball_in_play":1,"contact_window":0}}
{"path":"match_0001.npz","step":122,"events":{"contact_window":1}}
```

Supported events: `serve_ready`, `ball_in_play`, `contact_window`, `point_resolved`. An annotation step indexes an observation in the saved NPZ. Review and validate the dataset coordinate convention before using labels. Sparse annotations are allowed; unprovided labels are masked out of the corresponding loss.

## Training and inference

Edit the local ROM and Mesen paths in `configs/scenario.json` first. Example short emulator run:

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical --max-updates 3
```

Default hierarchy checkpoint paths are configured by `hierarchical.save_dir` and `hierarchical.log_dir` (defaults: `checkpoints_hierarchical_native256x240_frame1`, `runs_hierarchical_native256x240_frame1`). Evaluate a hierarchy checkpoint with:

```powershell
python -m tennis_rl.evaluate --config configs/scenario.json --checkpoint checkpoints_hierarchical_native256x240_frame1/hierarchical_latest.pt --episodes 20
```

For a software-only sanity test without Mesen/ROM:

```powershell
python scripts/sandbox_train_smoke.py
```

The sandbox smoke test uses generated observations and a mock environment. It validates finite optimizer updates, checkpoint round-trip, and a declining imitation loss on a synthetic easy task. It is not a real-game training run and cannot validate match win rate.

## State / reward correctness changes

- A start-segment timeout is treated as a service fault only if explicit scenario metadata identifies the agent as the server.
- Receiver timeouts are truncations with no service-fault penalty.
- GAE bootstraps from the final pre-reset observation for truncated episodes; the auto-reset observation is not used as the terminal bootstrap.
- The unvalidated centroid of image motion is disabled as a default auxiliary target.

## Evaluation protocol

Keep whole matches separated between training and hold-out. Do not split overlapping frames/windows from the same match across partitions. Use a fixed held-out set and report win rate plus confidence intervals and task-specific metrics (serve initiation/faults, return success, successful interceptions, shot outcomes, action duration, and wins by role/difficulty). Evaluate the flat-PPO baseline under the same hold-out and budget.

## Important limitations

- A Mesen executable, valid ROM, and actual savestate files are not bundled.
- Real-game throughput and win rate have not been measured by the repository's mock sandbox validation.
- No pre-existing model checkpoint or demonstrations are included in the provided archive.
- Auxiliary ball/player coordinate and event losses remain inactive unless explicit annotations are available.
- This architecture is a testable research implementation, not a claim of a validated winning policy. A falling synthetic imitation loss does not prove that the hierarchy improves at tennis.
