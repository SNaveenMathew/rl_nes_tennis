# Hierarchical Specialist Implementation — Sandbox Test Report

Date: 2026-10-18

## Scope

This report covers static/automated verification and small staged-training runs for the implemented hierarchical specialist pipeline. It does **not** claim the learned system plays NES Tennis well: the sandbox does not contain the user's Windows Mesen executable and could not download/launch an alternative emulator. No real-ROM episode, point, or match was completed in this sandbox.

## Implementation included

- Recurrent visual belief encoder and masked state/event prediction losses.
- Root phase gate, phase-conditional sub-gates, and thirteen specialist leaf policies.
- Top-down learned inference (root phase first, then branch-local selection) rather than a global argmax across leaves.
- Conditional movement/action output and explicit A/B pulse-duration labels; unknown pulse labels remain masked.
- Staged training for `foundation`, `perception`, `leaf`, `gates`, and `joint`, plus weight-compatible expert assembly.
- Task annotation and demonstration dataset audits, scenario counts, match-level split/extraction, task-stratified segment metadata, and per-scenario emulator evaluation.
- Corrected treatment of termination versus truncation, serve-role-specific start timeouts, recurrent-state tensor shape, and reward-to-go sourced only from actually recorded rewards.

## Automated repository tests

Command:

```powershell
python -m pytest -q
```

Result: **64 passed, 4 skipped** (6.73 seconds in this sandbox). The skipped cases depend on Lua embedding (`lupa`), which is not installed in this sandbox. These skipped tests still need to run on the Windows environment with the project's required dependencies installed.

The suite includes a regression test demonstrating that deterministic hierarchy inference must choose a root phase first and then choose a skill conditionally within that branch. It also covers unknown annotations/durations and guards against making segment length or point outcomes into fabricated local action labels.

Additional checks to run before packaging/release:

```powershell
python -m compileall -q tennis_rl tests
python -m tennis_rl.dataset stats --demo-root demonstrations
```

## Staged synthetic pipeline smoke

To verify that every new stage can execute without an emulator, a generated toy dataset was created outside the repository at `/mnt/data/tennis_rl_hierarchy_smoke`. It contains 65 artificial points, 39 train task segments, and 26 held-out segments, evenly spread over 13 artificial leaf tasks. Labels were intentionally simple, mostly constant by task, and must not be mistaken for human tennis demonstrations.

| Stage | Run | Observed result | Interpretation |
|---|---:|---|---|
| Foundation | 30 updates | finite losses; normalized ball-coordinate MAE 0.162 → 0.153; player MAE 0.193 → 0.181; toy event accuracy reached 1.0 | The optimizer and masked state losses execute. Artificial event labels make this metric unrepresentative of sprite tracking. |
| Independent `stroke_timing` leaf | 10 updates | finite loss; held-out action accuracy 1.0 on 24 toy labels; pulse accuracy 0.333 on 6 examples | Action labels are trivial in this dataset. Pulse performance is not good enough to claim timing has been learned. |
| Expert assembly | one independent leaf head | checkpoint assembly and shared-encoder compatibility check completed | Confirms the assembly path can load and combine compatible model weights, not that all experts are useful. |
| Gates | 30 updates | held-out phase accuracy peaked at 0.385 / 26; conditional leaf accuracy 0.308 / 26; full top-down skill accuracy reached 0.231 / 26 | Weak and noisy on an intentionally tiny artificial holdout. The global/tactical discriminations are not shown to be adequate. |
| Perception-only fine-tune | 3 updates | finite losses; ball MAE 0.152 → 0.150, player MAE 0.181 → 0.179 | Smoke-check only; too few updates and toy labels. |
| Joint staged fine-tune | 3 updates | finite losses; phase accuracy about 0.385–0.462 / 26; leaf accuracy 0.308 / 26; full manager skill accuracy 0.231 / 26 | All intended stage code paths execute. This does not establish robust routing or playing strength. |

The synthetic dataset is not bundled with the repository. It is excluded to prevent toy data from being mistaken for real demonstrations.

## General mock-RL smoke test

Command:

```powershell
python scripts/sandbox_train_smoke.py
```

Result: 3 mock hierarchical PPO updates completed with finite losses (`0.236`, `0.165`, `0.226`), and checkpoint save/load round-trip succeeded. The supervised sanity task reduced its synthetic behavior-cloning loss from `4.6711` to `2.7887` over ten updates (about 40.3%). PPO loss need not decrease monotonically; this sanity check validates optimization and checkpoint plumbing only. It does not measure a real policy's point win rate.

## Segment extraction, replay, and reward-estimation contract

- Demonstration NPZ files remain the source of numeric observations, actions, raw rewards, episode boundaries, and actual point outcomes.
- JSON sidecars contribute explicit scenario tags and recorder metadata; dataset statistics report tag counts and pulse-label coverage.
- Task annotation intervals are half-open `[start, end)`, validated against point lengths. An explicitly known leaf task supervises that leaf; broad/legacy phase labels supervise only the phase gate.
- `manager_duration` is a separate explicit annotation. It is not inferred from interval length. Empty duration remains unknown and is masked from duration supervision.
- Unknown A/B pulse lengths remain unknown (`-1` in the training data) rather than being assigned a guessed pulse.
- The segment catalog records source reference, split, task/phase labels, interval, and discounted return from rewards actually recorded in that interval. It does not invent a reward for an action merely because the point eventually won, nor assume every action in a lost point was bad.
- Train/holdout split is keyed by match to prevent overlapping windows from the same match crossing the split.
- Evaluation treats unresolved/truncated point outcomes as unknown, not wins or losses. For valid comparisons, use the same savestate manifest, split, repeat count, random seed and episode horizon.

Human annotation and external game-state validation are still necessary. No ground-truth ball position or contact event is inferred from the former pixel-motion-centroid heuristic.

## MesenCE 2.2.1

The repo config and guide now target the official MesenCE 2.2.1 Windows build. The upstream release is newer than 0.9.9 and notes a 2.2.1 hotfix for iNES ROM handling. Use the official ZIP, extract it, and point `mesen.executable` to the extracted `Mesen.exe`. The official Windows ZIP SHA-256 recorded in the current Scoop metadata is:

`c19e10b3b3865eb0e86e4d01bf09a5027a4452081c04b5bd5b6cbbf76e346fd6`

Official release: <https://github.com/nesdev-org/MesenCE/releases/tag/2.2.1>

The Linux sandbox did not have MesenCE 2.2.1 installed and could not fetch GitHub assets due network DNS restrictions. Therefore `--testrunner`, Lua permissions, lock-step file-mailbox IPC, real scenario restoration, and actual ROM reward/score telemetry were **not** integration-tested here. Run `scripts/check_mesen.ps1` on Windows before training; it launches a one-worker real-emulator smoke test.

## Promotion rule

Do not promote a specialist because its loss decreases or because it has high accuracy on constant toy labels. Promote only after held-out human-labeled scenarios show the relevant skill metric improves, paired full-point evaluations show no regression, and results are reported separately for serve/receive roles with enough completed points. For early screening, target at least 100 completed points per checkpoint; for steadier comparisons use 300–500 points and multiple seeds when practical.
