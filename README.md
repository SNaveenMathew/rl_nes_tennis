# Tennis RL

Reinforcement learning framework for training an autonomous agent on NES Tennis using headless Mesen / MesenCE emulation, recurrent Proximal Policy Optimization (PPO), and lock-step atomic file-mailbox inter-process communication.

## Architecture

- **Emulator Transport & Bridge (`mesen/bridge.lua`, `tennis_rl/mesen_client.py`)**:
  - Headless Mesen execution using `--testrunner` mode.
  - Zero network dependencies (no LuaSocket required); uses standard Lua I/O with atomic file-mailbox IPC.
  - Framebuffer grayscale capture (84x84 downscaled) directly from screen buffer.
  - Diagnostic RAM inspection for game score tracking and sparse/dense reward calculation.
- **Parallel Vectorized Environment (`tennis_rl/env.py`)**:
  - Batched multi-worker stepping using `ThreadPoolExecutor`.
  - Frame stacking (4 frames by default) and automatic episode reset handling.
  - Robust worker fault isolation and transparent hard-restart recovery.
- **Neural Network Policy (`tennis_rl/model.py`)**:
  - Visual convolutional feature extractor (`VisualEncoder`).
  - Recurrent GRU core (`RecurrentActorCritic`) for temporal state modeling across serves, ball trajectories, and court positioning.
  - Orthogonally initialized discrete actor and critic value heads.
- **Recurrent PPO Algorithm (`tennis_rl/ppo.py`)**:
  - Generalized Advantage Estimation (GAE).
  - Sequence chunking for recurrent BPTT updates.
  - Clipped surrogate objective with value function clipping.
  - Mixed-precision training (`torch.amp`).
  - TensorBoard metric logging and automatic checkpoint management.

## Project Structure

```
tennis_rl_production/
├── configs/
│   └── base.json           # Default training & environment hyperparameters
├── mesen/
│   ├── bridge.lua          # Lua bridge script for Mesen emulator
│   └── README.md           # Mesen bridge documentation
├── scripts/
│   ├── train.ps1           # Training launcher script
│   ├── evaluate.ps1        # Evaluation script
│   └── smoke_test.ps1      # Bridge smoke test script
├── tennis_rl/
│   ├── __init__.py
│   ├── actions.py          # Action space mappings (19 controller combinations)
│   ├── benchmark_env.py    # Environment throughput benchmarking tool
│   ├── config.py           # Typed dataclass configuration schemas
│   ├── env.py              # Vectorized parallel environment with fault tolerance
│   ├── evaluate.py         # Policy evaluation against game AI
│   ├── mesen_client.py     # Mesen worker process management & IPC mailbox
│   ├── model.py            # Visual CNN + GRU Recurrent Actor-Critic network
│   ├── ppo.py              # Recurrent PPO trainer with GAE and AMP
│   └── train.py            # Main training loop & checkpoint management
├── tests/                  # Contract, protocol, model, and action test suite
├── tools/
│   └── calibrate_state.py  # Utility for capturing starting savestates
├── Makefile                # Test discovery and compilation targets
├── requirements.txt        # Runtime dependencies
└── LICENSE                 # MIT License
```

## Quick Start

### Installation

Ensure Python 3.10+ (or Python 3.14 with PyTorch CUDA support) is installed:

```bash
pip install -r requirements.txt
```

### Configuration

Edit `configs/base.json` to specify your local Mesen executable and NES Tennis ROM paths:

```json
{
  "mesen": {
    "executable": "path/to/Mesen.exe",
    "rom": "path/to/Tennis (JU).nes"
  }
}
```

### Smoke Test

Verify the emulator bridge and worker IPC:

```powershell
python -m tennis_rl.train --config configs/base.json --smoke-test
```

### Training

```powershell
python -m tennis_rl.train --config configs/base.json
```

Or resume from a checkpoint:

```powershell
python -m tennis_rl.train --config configs/base.json --resume checkpoints/tennis_latest.pt
```

### Evaluation

```powershell
python -m tennis_rl.evaluate --config configs/base.json --checkpoint checkpoints/tennis_latest.pt --episodes 20
```

### Benchmarking

Measure throughput of the parallel emulator environment:

```powershell
python -m tennis_rl.benchmark_env --config configs/base.json --seconds 10.0
```

### Testing

Run the test suite:

```bash
pytest -v
```
