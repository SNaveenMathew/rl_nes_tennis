# NES Tennis RL — strict replay test shutdown fix

## What changed

The latest Windows trace showed that all replay frame assertions succeeded (`LOAD_FROM 7` produced `[base + 7] * 3`, `LOAD_FROM 0` produced `base + 1`, and all action gaps matched), but the test then failed with a strict-lockstep host-command timeout.

This was a test-lifecycle bug, not a frame-count mismatch: after its *last* observation, the strict test host finished without sending another command. The bridge correctly remained parked waiting for the next command and then failed closed at its configured 0.701-second deadline. That timeout was expected under the protocol but should not be treated as a replay error when a test intentionally ends.

`tests/test_state_ops_test.py` now sends an explicit `Q` from its strict-mode cleanup path. Its emulator-driving loop continues until the Lua bridge consumes `Q` and calls `emu.stop(0)`, so the test does not return early and leave a race. The test asserts that the shutdown command was sent and consumed, and still asserts that `error.txt` is empty. The strict timeout and frame determinism assertions are not relaxed. A source-level regression test now protects this shutdown behavior even in environments where `lupa` is absent.

## Validation performed

- `python -m pytest -q`: **69 passed, 4 skipped**.
- `python -m compileall -q tennis_rl tests scripts tools`: passed.
- `git diff --check`: passed.
- Native Lua 5.4 ctypes harness, using the repository's actual `mesen/bridge.lua` and `tests/mock_emu.lua`: **20 consecutive full replay runs passed** with graceful shutdown enabled. Every run produced base frame 12, three `LOAD_FROM 7` frames at 19, `LOAD_FROM 0` at 13, four-frame action gaps, four state loads, `emu.stop(0)`, and an empty bridge error file.
- Clean archive extraction and archive integrity: passed.

## Limitation

Four integration modules use `pytest.importorskip("lupa")` and were skipped in this Linux sandbox because the `lupa` package is unavailable and package downloads are blocked. The native Lua test validates the same bridge script and mock emulator behavior, but does not substitute for running the complete Lupa-backed test on Windows/Mesen. The current change directly fixes the shutdown issue shown in the Windows trace; a final Windows pytest run is still required to confirm the Lupa integration path.

## Windows test command

After extracting this ZIP into a fresh directory and activating the project's `tennis_rl` environment:

```powershell
python -m pytest -q
```
