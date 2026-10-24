# Sandbox validation report

**Date:** 2026-10-19 (Mesen callback and framebuffer retry update)
**Scope:** Static checks, unit/mock integration tests, and synthetic training smoke test. No actual NES gameplay was run.

## Results

| Check | Result |
|---|---|
| `python -m compileall -q tennis_rl tests scripts` | Passed |
| `python -m pytest -q` | **84 passed, 5 skipped** (7.68 seconds) |
| Config parse (`configs/base.json.example`, `configs/scenario.json`) | Passed; hierarchical pipeline configured with 13 specialized tasks and 27 actions; new policy observations are 256×240 grayscale |
| `python scripts/sandbox_train_smoke.py` | Passed; three mock hierarchical PPO updates, finite metrics, checkpoint save/load round-trip; synthetic BC loss 4.6703 → 2.8291 |
| Synthetic supervised sanity task | Passed; explicit behavior-cloning loss fell from **6.0044** to **4.1783** over 10 updates (about **30.4% lower**) |
| `git diff --check` | Not run: this exported source directory has no `.git` metadata |

The skipped tests are Lupa-dependent and are skipped here because `lupa` is not installed in this sandbox. The bridge and mock were additionally exercised with the system Lua 5.4 shared library through `ctypes`; that is useful mock coverage but does not replace running the Lupa tests and live Mesen on Windows.

## Latest recording/annotation runtime contract

- New policy observations: four stacked **256×240 grayscale** frames; default `mesen.frame_skip=1` (approximately 60.1 decisions per second). Use the same spatial dimensions and frame skip for demonstrations, training and inference.
- When recording is enabled, `capture_native_frames=true` stores aligned native **256×240 RGB** frames in each point NPZ for annotating sprites/events; training continues to consume grayscale policy frames. Native frames are not synthesized by enlarging old 84×84 or 128×120 data.
- `scripts/record.ps1` defaults to `configs/scenario.json`; `scripts/reset_demonstrations.ps1` is a guarded, explicit clean-start command. `scenario_states` is only removed when the operator deliberately adds `-IncludeScenarioStates`.
- `python -m compileall -q tennis_rl tests scripts`: passed. CLI `--help` smoke checks passed for training, staged training, scenario evaluation, annotation, and dataset tools.
- The Python suite skips Lupa-dependent tests because `lupa` is not installed here. The actual Windows Mesen 2.2.1 process and ROM were not run, so follow `scripts/check_mesen.ps1` before bulk recording.

## Three mock RL updates

The smoke script collected transitions from a deterministic mock worker and completed three optimizer updates. Total loss by update was approximately `0.3846`, `0.4433`, and `0.1869`; manager and expert sample counts were nonzero and all reported metrics were finite. PPO total loss is not expected to decline monotonically, so the loss sequence is not used as a claim of policy improvement.

The separate synthetic demonstration task used identical blank observations with explicit `wait`/neutral-action labels. Its imitation loss declined monotonically:

`6.0044 → 5.8843 → 5.7465 → 5.5854 → 5.4004 → 5.1922 → 4.9628 → 4.7149 → 4.4520 → 4.1783`

This validates that gradients can learn a simple, supplied synthetic label. It does not demonstrate correct shot timing, tennis tactics, or improved match outcomes.

## Regression coverage added

- Truncated GAE bootstraps with the final pre-reset observation, rather than the next episode's reset observation.
- A start-phase receiver timeout is a truncation and does not receive the server's service-fault penalty.
- An equivalent server timeout is marked as a service fault and receives the configured penalty.
- Hierarchical actor/critic forward pass, action shapes, manager/controller flow, trainer update, checkpoint round-trip, and mock rollout collection.
- Task annotations are read only when explicitly provided; path traversal outside the demonstration directory is rejected.
- Existing segment replay tests use generated temporary demonstrations rather than assuming an absent local `demonstrations/` directory.

## What remains unvalidated

The supplied archive includes no NES ROM, Mesen executable, saved scenario states, recorded demonstration NPZ files, or real-game checkpoint. Therefore:

- No real-emulator training iterations were possible in this sandbox.
- No held-out tennis win rate or real-game learning curve was measured.
- The hierarchy is a runnable research implementation, **not yet a validated winning NES Tennis model**.

To establish real improvement, configure your installed Mesen and ROM paths, record/harvest scenario states, run several actual-emulator updates, and compare fixed held-out match results against the preserved flat-PPO baseline. Do not use training-point win rate alone as evidence of generalization.

## State-ops determinism follow-up (2026-10-19)

A Windows run exposed that a recording-mode environment variable leaked from one embedded-Lua test into the following replay test. The tests now inject their per-test `MESEN_RL_*` variables explicitly, rather than relying on process environment mutation, and the state replay assertions run in strict lockstep. `MESEN_RL_STRICT_LOCKSTEP=1` is optional and fails closed on an overdue host command so a deterministic replay cannot silently drift; the default training mode remains unchanged.

Verification after the correction: Python test suite **79 passed, 4 skipped**; the skips are Lupa-dependent because `lupa` could not be installed in this network-isolated sandbox. A separate system-Lua 5.4 `ctypes` harness ran the bridge and mock emulator through 20 replay repetitions, all passing exact frame checks (`base=12`, `loads=[19,19,19]`, `zero=13`, step gaps `[4,4,4,4]`). This does not replace the required live Windows/MesenCE smoke test.

## Test harness guard for legacy bridge diagnostics (2026-10-19)

Fixed the latest reported `FileNotFoundError: ... api_info.txt` in `tests/test_state_ops_test.py`. Only the current bridge writes `api_info.txt`; the old-bridge fixture intentionally does not. The harness now checks that the launched bridge is the current `mesen/bridge.lua` before reading/validating that diagnostic file. The required-file assertion remains for the current bridge, so a genuine missing diagnostic still fails clearly.

After this change: `python -m compileall -q tennis_rl tests` passed; `python -m pytest -q` reported **79 passed, 4 skipped**. Lupa-dependent tests are skipped in this sandbox (module unavailable), so the user's Windows/Lupa run is required to confirm the exact embedded-Lua test path. Live Mesen/ROM integration is not claimed.

## Strict-lockstep test teardown regression (2026-10-19)

The latest reported error was a strict-lockstep timeout after the determinism results had already matched all expectations. The strict bridge was waiting for a command after the test host had completed its final observation. The test teardown now sends the protocol's `Q` command and continues driving the mock until the bridge consumes it (`emu.stop(0)`); it asserts clean shutdown and leaves strict timeout behavior unchanged for real host stalls.

Updated validation: `python -m compileall -q tennis_rl tests scripts` passed; `python -m pytest -q` reports **80 passed, 4 skipped**. The four Lupa-backed tests are skipped in this sandbox because the Lupa binding is unavailable. A native Lua 5.4 `ctypes` run of the actual bridge and mock emulator passed **20/20** strict replay repetitions with `base=12`, `loads=[19,19,19]`, `zero=13`, `gaps=[4,4,4,4]`, `emu.stop(0)`, zero timeout hits, and empty error files. This is a test-mock check, not a live MesenCE/ROM integration result.


## Mesen callback and framebuffer regression follow-up (2026-10-19)

The user's real-emulator smoke test exposed two bridge defects: savestate load/save was attempted outside the required CPU `exec` memory callback, and capture crashed if `getScreenBuffer()` was nil/empty during startup or restore. The bridge now reads the current main CPU PC using `emu.getState().cpu.pc` (with a compatibility fallback), arms the savestate operation at that instruction address with `emu.callbackType.exec`, then removes the completed callback from an `endFrame`/`startFrame` event—not while Mesen is iterating memory callbacks. This also avoids a callback on every CPU instruction.

Framebuffer capture now validates the table/index layout, accepts zero- or one-based ARGB arrays, waits/retries a pending observation for up to 180 frames if the buffer is missing/incomplete, and reports a clear bridge error if it never becomes valid. Native annotation frames keep a fixed 256×240 RGB payload; when the emulator supplies 239 rows, the final output row duplicates the final visible row. `configs/scenario.json` is aligned with `C:/Games/NES/Mesen2.2.1/Mesen.exe` and the ROM path shown in the user's log.

Latest checks after these changes:

- `python -m pytest -q`: **84 passed, 5 skipped** in 7.68 seconds in this environment. The new capture integration module is skipped here with the Lupa-dependent tests because Lupa is unavailable.
- System Lua 5.4 `ctypes` harness executed the actual bridge + `tests/mock_emu.lua`: an initially unavailable screen buffer was retried and produced the expected 8×8 grayscale observation; `SAVE_TO` then `LOAD_FROM 7` landed exactly seven frames after the saved frame and both quit cleanly with zero memory callbacks left installed.
- A second system-Lua run tested native RGB capture with four initial nil framebuffer reads and a 239-row source; it emitted exactly 8×8 grayscale bytes plus 256×240×3 RGB bytes, duplicated the final visible row, and stopped cleanly with no bridge/callback errors.

**Still not verified:** live Mesen/ROM integration on Windows. These harness checks validate the Lua protocol against the mock only. Re-extract the new ZIP over the working project and rerun `scripts/check_mesen.ps1` before recording.

Repeat validation: the complete strict replay protocol passed **10/10** repetitions with `LOAD_FROM 7` consistently landing at base+7, `LOAD_FROM 0` landing at base+1, four-frame action gaps, no callback leaks, zero exit code, and empty error logs. The framebuffer retry/save-load and 239-row native-RGB checks also passed in **10/10** repetitions each. These runs use the actual Lua bridge plus the repository mock via system Lua 5.4; they are not live Mesen/ROM runs.
