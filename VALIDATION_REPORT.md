# Sandbox validation report

**Date:** 2026-10-18  
**Scope:** Static checks, unit/mock integration tests, and synthetic training smoke test. No actual NES gameplay was run.

## Results

| Check | Result |
|---|---|
| `python -m compileall -q tennis_rl tests scripts` | Passed |
| `pytest -q` | **60 passed, 4 skipped** |
| Config parse (`configs/base.json.example`, `configs/scenario.json`) | Passed; hierarchy enabled with 7 default skills and 27 actions |
| `python scripts/sandbox_train_smoke.py` | Passed; three mock hierarchical PPO updates, finite metrics, checkpoint save/load round-trip |
| Synthetic supervised sanity task | Passed; explicit behavior-cloning loss fell from **6.0044** to **4.1783** over 10 updates (about **30.4% lower**) |
| `git diff --check` | Passed |

The four skipped tests exercise the Lua/Mesen bridge through `lupa`. `lupa` is listed in `requirements.txt`, but it was not installed in the sandbox and package-network access was unavailable, so those integration tests could not be executed here.

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
