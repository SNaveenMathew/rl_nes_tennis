# Hierarchical Specialist Implementation — Sandbox Test Report

Date: 2026-10-19 (Mesen callback and framebuffer retry update)

## Latest verification — clean demonstration rebuild and native capture

The current package changes the new demonstration contract to four stacked **128×120 grayscale** policy frames per step and optionally stores aligned **256×240 RGB** native frames for human review/annotation. Recording defaults to `frame_skip=2` (about 30 policy decisions/second for a ~60.1 Hz NES frame clock); recording, training and inference must use the same frame skip and spatial shape. Native RGB storage is only enabled in record mode and is sent as an appended payload after the grayscale observation, so normal policy observations remain compact.

The added clean-start helper is `scripts/reset_demonstrations.ps1`. It asks the operator to type `RESET DEMONSTRATIONS` before deleting generated demonstration and review outputs. `scenario_states` is excluded unless `-IncludeScenarioStates` is passed. It does not delete the repository code, ROM, emulator or checkpoints. The recording launcher defaults to `configs/scenario.json`. Annotation guidance documents native RGB contact sheets/GIFs, optional explicit task/state labels, half-open task intervals, and a review-only `--sample-every` rate that does not change game timing.

Latest checks on 2026-10-19:

- `python -m compileall -q tennis_rl tests scripts`: passed.
- `python -m pytest -q`: **84 passed, 5 skipped** (6.90 seconds). The skips are Lupa-dependent; the Python `lupa` binding is unavailable in this sandbox.
- CLI help smoke checks for `tennis_rl.train`, `tennis_rl.staged_train`, `tennis_rl.evaluate_scenarios`, `tennis_rl.annotate_demonstrations`, and `tennis_rl.dataset`: passed.
- `python scripts/sandbox_train_smoke.py`: passed; three synthetic mock-RL updates completed with finite losses, checkpoint save/load round-trip passed, and the synthetic behavior-cloning loss fell from about 4.6703 to 2.8291. These toy metrics are plumbing checks only, not gameplay evidence.

**Not verified here:** a live Windows run using MesenCE 2.2.1 and the user's actual Tennis ROM. Run `scripts/check_mesen.ps1` on the Windows machine before collecting a full dataset. No real-game win-rate claim is made.


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

Result after the Windows savestate-path fix: **65 passed, 4 skipped** (7.40 seconds in this sandbox). The skipped cases depend on Lua embedding (`lupa`), which is not installed in this sandbox. These skipped tests still need to run on the Windows environment with the project's required dependencies installed.

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

## Windows savestate-path regression fix (user-reported pytest failure)

The reported failures in `tests/test_record_mode_test.py` and `tests/test_state_ops_test.py` showed `io.open(..., "wb")` rejecting a `SAVE_TO` target whose name ended with a Windows-style mixed path and a hidden carriage return. The tests write commands with Python text-mode `Path.write_text()` on Windows (CRLF), while the Lua bridge opens `command.in` in binary mode. Lua can therefore return a command line ending in `\r`; because `SAVE_TO` captures the rest of the line as a path, that CR can become part of the filename. Token-based commands can appear normal, which made this failure easy to miss.

The bridge now strips a trailing carriage return from every mailbox command before parsing it, normalizes the IPC directory and explicit `SAVE_TO` / `LOAD_FROM` paths to forward slashes, and has a source regression test for these safeguards. This keeps the command protocol compatible with both LF-only byte writes from `MesenWorker` and CRLF commands from Windows text-mode test/helper code.

Validation in this Linux sandbox: `python -m compileall -q tennis_rl tests` passed; `python -m pytest -q` passed **65 tests, 4 skipped**. The skipped Lua-embedding tests could not execute here because `lupa` is not installed. The two user-reported tests should be rerun on Windows with `lupa` installed; I have not claimed a direct Windows/Mesen integration pass from this sandbox.

## MesenCE 2.2.1

The repo config and guide now target the official MesenCE 2.2.1 Windows build. The upstream release is newer than 0.9.9 and notes a 2.2.1 hotfix for iNES ROM handling. Use the official ZIP, extract it, and point `mesen.executable` to the extracted `Mesen.exe`. The official Windows ZIP SHA-256 recorded in the current Scoop metadata is:

`c19e10b3b3865eb0e86e4d01bf09a5027a4452081c04b5bd5b6cbbf76e346fd6`

Official release: <https://github.com/nesdev-org/MesenCE/releases/tag/2.2.1>

The Linux sandbox did not have MesenCE 2.2.1 installed and could not fetch GitHub assets due network DNS restrictions. Therefore `--testrunner`, Lua permissions, lock-step file-mailbox IPC, real scenario restoration, and actual ROM reward/score telemetry were **not** integration-tested here. Run `scripts/check_mesen.ps1` on Windows before training; it launches a one-worker real-emulator smoke test.

## Promotion rule

Do not promote a specialist because its loss decreases or because it has high accuracy on constant toy labels. Promote only after held-out human-labeled scenarios show the relevant skill metric improves, paired full-point evaluations show no regression, and results are reported separately for serve/receive roles with enough completed points. For early screening, target at least 100 completed points per checkpoint; for steadier comparisons use 300–500 points and multiple seeds when practical.

## Follow-up: legacy savestate regression harness

The user's next Windows run showed that the current bridge now successfully saved and loaded the new file-based savestate (`STATE|...|binarytail`, two loads, no bridge error), but collection still failed while probing the archived Mesen 0.9.x bridge. That probe incorrectly called the **new** lockstep `step()` helper against the old bridge fixture, which writes the older observation protocol and therefore does not create `observation_ready_<token>.ready`. The failed helper thread never populated `acked`, causing a secondary `KeyError` that obscured the real issue.

The test now sends only the legacy `SV` request when checking the legacy fixture's unsupported savestate API. Its purpose is narrowly to assert that the old API failure is reported clearly; it does not run a current-protocol observation step against an old-protocol fixture. The harness also signals thread completion in a `finally` block and surfaces helper-thread assertion failures promptly instead of waiting up to 40 seconds and failing later with missing dictionary fields.

Sandbox verification after this change: `python -m pytest -q` reports **65 passed, 4 skipped**. The four Lua-dependent tests are skipped because `lupa` is unavailable in this sandbox, so the exact Lua integration case has **not** been executed here. Please rerun the full command on Windows; the expected result is that `test_state_ops_test.py` gets beyond the legacy-protocol probe and completes its new-bridge determinism checks.


## Follow-up: nondeterministic scenario replay frame count

The next Windows report showed that saved-state writes/loads were now successful but repeated `LOAD_FROM 7` trials saved at different frames. Root cause: the bridge acknowledged completion of its requested no-op horizon from `endFrame`, then returned to the main callback. Because the host can only post the next command asynchronously, lockstep timeout slices could let additional idle frames elapse before the next `SAVE_TO` was consumed. This made scenario replay depend on host scheduling latency.

The bridge now treats the end of the requested no-op horizon as a lockstep barrier: it acknowledges the load, waits for the next command in the same `endFrame` callback, and returns before the current frame can be reused as an action/hold frame. A following state operation executes at the next `startFrame`; a newly received action starts counting on the next actual frame. The regression assertions have been strengthened: three `LOAD_FROM 7` + `SAVE_TO` trials must save at exactly `base_frame + 7`, and `LOAD_FROM 0` must save at `base_frame + 1` under the documented mock protocol.

Because the Python `lupa` package is not installed in this sandbox and package network access is unavailable, I exercised the real `mesen/bridge.lua` and existing `tests/mock_emu.lua` directly through the system Lua 5.4 shared library using a small external `ctypes` host harness that reproduced the command/ack sequence. Five independent runs returned `base=12`, `loads=[19,19,19]`, and `zero=13`; the exact assertions passed in all five runs. This is a Lua 5.4 mock-emulator regression run, not a MesenCE GUI/ROM integration test. The regular Python suite and source-level barrier regression are run separately and the Windows Lupa-backed determinism test should be rerun on the user's machine.


## Follow-up: stable acknowledgement for `LOAD_FROM 0` and token reuse

The replay barrier is now applied consistently to both `LOAD_FROM 0` and `LOAD_FROM N` where N is positive. All file-based `LOAD_FROM` operations set `noop_ack_pending`; `needs_frames()` remains true through the replay horizon; and the bridge writes `state_loaded.txt` only at the end-frame boundary. It then waits for the next host command before returning from that callback. This avoids a race in which `LOAD_FROM 0` could be acknowledged during `startFrame` and allow extra frames to elapse while the host issued the next command. It also avoids counting the terminal replay frame as part of an action that was only received after that frame's input poll. The state reset clears action/pulse and held-button tracking before replayed frames are simulated.

The determinism test now asserts exact outcomes rather than a weak `loads`-are-equal condition: three `LOAD_FROM 7` runs must save at `base + 7`, `LOAD_FROM 0` must save at `base + 1` in the mock callback model, and the four-frame action-step gaps must all be equal. The `step()` helper deletes a previous `observation_ready_<token>.ready` before reusing a token, preventing an old marker from falsely satisfying the next wait.

Validation in this sandbox:

- `python -m compileall -q tennis_rl tests`: passed.
- `python -m pytest -q`: **66 passed, 4 skipped**. The skipped tests are protected by `pytest.importorskip("lupa")` because this sandbox does not have the Python `lupa` binding and package-network access is unavailable.
- An external `ctypes` harness executed the repository's actual `mesen/bridge.lua` and `tests/mock_emu.lua` using the system Lua 5.4 shared library and reproduced the restore/step protocol in five independent runs. Every run produced `base=12`, `loads=[19,19,19]`, `zero=13`, `gaps=[4,4,4,4]`, and `loaded=4`; the exact assertions passed all five times. This covers the Lua bridge logic against the repository mock, but is not a live MesenCE/ROM integration test.

Please rerun `python -m pytest -q` on Windows. The expected result is that the state-operations determinism check no longer reports divergent load frames; the actual Windows/Lupa suite and MesenCE 2.2.1 integration still need confirmation on the target laptop.

## Follow-up: replay test failed when record-mode environment leaked into other Lua tests

The latest Windows traceback (`base=213`, `loads=[280,381,258]`) was caused by test-environment contamination, not evidence that the requested seven no-op frames themselves were varying. `test_record_mode_test.py` deliberately sets `MESEN_RL_RECORD_MODE=1`; the old state-ops harness only called `os.environ.update(...)`, leaving that value visible to the embedded Lua runtime. Since record mode disables lockstep, the replay test could advance arbitrarily while the Python host thread waited. Embedded Lua runtimes on Windows also do not necessarily read the same environment snapshot as Python, so the tests now inject an explicit `MESEN_RL_*` environment mapping into each Lua runtime.

The bridge also now supports optional `MESEN_RL_STRICT_LOCKSTEP=1` for deterministic replay/evaluation. In this mode, it waits for the next host command at a frame boundary and stops with an explicit error if the strict wait deadline expires; it does not silently run timeout-driven idle frames that would invalidate exact replay comparisons. Normal training keeps the existing sliced/soft lockstep behavior by default. The Lupa regression test enables strict mode only for the exact state-restore accounting assertions and yields the Python scheduler from its test-only hook so the embedded Lua callback does not starve its simulated host thread.

Validation after this correction:

- `python -m py_compile` on the modified Lua regression tests and helper: passed.
- `python -m pytest -q`: **79 passed, 4 skipped**. The four skipped cases need the Python `lupa` binding, unavailable in this sandbox.
- A separate `ctypes` harness executed the current `mesen/bridge.lua` and `tests/mock_emu.lua` through the system Lua 5.4 shared library for **20 consecutive replay runs**. All runs produced `base=12`, `loads=[19,19,19]`, `zero=13`, and `gaps=[4,4,4,4]`; every exact assertion passed.

This validates the replay protocol under the project mock. The Windows/Lupa test suite and live MesenCE + ROM still need to be run on the target laptop; neither is claimed as a sandbox integration pass.

## Follow-up: legacy bridge test incorrectly required `api_info.txt` (2026-10-19)

A subsequent Windows traceback showed `FileNotFoundError` for `api_info.txt` while `run()` was loading `tests/fixtures/original_bridge.lua`. That fixture emulates the original bridge and intentionally does not publish the diagnostics file written by the current `mesen/bridge.lua`. The harness was reading `api_info.txt` before checking which bridge had been launched. The read now occurs only in the current-bridge branch, with an explicit existence assertion there. This preserves the configuration/strict-lockstep check for the new bridge while allowing the legacy fixture's separate `SV` failure test to exercise its actual contract.

Verification in this sandbox after the test-harness fix: `python -m compileall -q tennis_rl tests` passed and `python -m pytest -q` reported **79 passed, 4 skipped**. The 4 Lupa-dependent bridge tests remain skipped here because the Python `lupa` module is unavailable; this means the changed branch was inspected but not executed with Lupa in this sandbox. Please rerun the full suite in the Windows `(tennis_rl)` environment, where Lupa is installed, and run `scripts/check_mesen.ps1` for the live emulator check.

## Follow-up: strict-lockstep replay test must shut down cleanly (2026-10-19)

The user's latest Windows traceback occurred **after** the exact replay values passed: `LOAD_FROM 7` produced `[19, 19, 19]` from base frame `12`, `LOAD_FROM 0` produced `13`, and all four observed action gaps were `4`. The bridge's strict lockstep then correctly remained parked waiting for the next host command. Because the test script had ended after its final observation without sending another command, the bridge hit its strict timeout and wrote an error; that expected end-of-test idle timeout was incorrectly treated as a replay failure.

The test harness now sends `Q` in the strict test's `finally` path, keeps advancing the mock until `Q` is consumed and `emu.stop(0)` is observed, and asserts that graceful shutdown was sent and consumed. The strict timeout itself and exact frame-count assertions are unchanged, so a genuinely stalled host still fails closed. A source-level regression test protects the shutdown contract even when Lupa-dependent tests are skipped.

Validation in this sandbox:

- `python -m compileall -q tennis_rl tests scripts`: passed.
- `python -m pytest -q`: **80 passed, 4 skipped**.
- Native Lua 5.4 `ctypes` harness using the repository's actual `mesen/bridge.lua` and `tests/mock_emu.lua`: **20 consecutive strict replay runs passed**. Each recorded base `12`, `loads=[19, 19, 19]`, `zero=13`, `gaps=[4, 4, 4, 4]`, frame `42`, four state loads, `emu.stop(0)`, zero Lua timeout hits, and an empty `error.txt`.

The four Lupa integration tests remain skipped here because Lupa is not installed and package downloads are blocked. The native Lua check exercises the actual bridge and mock emulator behavior, but the Windows `(tennis_rl)` pytest run is still required to confirm the embedded-Lua integration end to end. Live MesenCE/ROM behavior is not claimed as tested.


## Mesen 2.2.1 smoke-test fixes (2026-10-19)

The user's `scripts/check_mesen.ps1` output identified a runtime integration failure despite the previous mock suite passing. The bridge now schedules savestate operations in an address-specific CPU `exec` callback because Mesen requires `createSavestate()` / `loadSavestate()` to run in that callback context. The bridge disarms the callback at a frame event, rather than removing a memory callback while Mesen iterates its callback collection. It also retrieves the current PC from `emu.getState().cpu.pc` before scheduling, minimizing the callback to one instruction address.

The screen capture path now treats nil, empty, sparse, or temporarily unavailable `getScreenBuffer()` arrays as transient. It validates the pixel array layout before sampling and retries the pending observation for up to 180 frames. It accepts zero-based and one-based arrays and keeps native annotation frames at a stable 256×240 RGB size when 239 visible rows are exposed.

Verification performed in the sandbox:

- Python suite: **84 passed, 5 skipped**; skips are Lupa-dependent because the Python Lupa binding is not installed here.
- Native Lua 5.4 harness: frame capture passed after four initial nil framebuffer results; `SAVE_TO`/`LOAD_FROM 7` returned the frame counter exactly seven frames after the saved state; `emu.stop(0)`, no lingering memory callbacks, and no bridge errors.
- Native RGB harness: four nil results followed by a 239-row ARGB buffer produced exactly 256×240×3 native RGB bytes, with row 239 padded from row 238, and no errors.

These remain mock-based protocol checks. The new ZIP must still be extracted on the Windows host, and `scripts/check_mesen.ps1` must be rerun against the actual Mesen executable and ROM before bulk recording or training.


The current native Lua stress validation was repeated **10/10 times** for the complete strict replay scenario (`LOAD_FROM 7` three times, `LOAD_FROM 0`, four action steps, clean `Q` teardown), plus **10/10 times** for nil-framebuffer recovery, save/load, and 239-row native RGB padding.
