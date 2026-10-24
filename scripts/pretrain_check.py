"""Pre-training system & pipeline validation for tennis_rl.
Verifies all dependencies, CUDA GPU, emulator/ROM paths, demonstration dataset,
scenario catalog, model initialization, segment replayer, and checkpoint directories.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tennis_rl.config import load_config
from tennis_rl.demonstrations import DemonstrationDataset, summarize as summarize_demos
from tennis_rl.dqfd import DQFDTrainer, RecurrentDuelingQ, Sequence
from tennis_rl.model import RecurrentActorCritic
from tennis_rl.scenarios import load_manifest
from tennis_rl.scenario_sampler import ScenarioLibrary, ScenarioSampler, SamplerConfig
from tennis_rl.segment_replay import DemoSegmentReplayer, SegmentReplayConfig


def run_checks() -> bool:
    all_ok = True
    print("=" * 70)
    print(" NES TENNIS RL - PRE-TRAINING VERIFICATION CHECKLIST")
    print("=" * 70)

    # 1. Environment & CUDA
    print("\n[1/6] Checking Python Environment & Hardware...")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version:    {sys.version.split()[0]}")
    print(f"  PyTorch version:   {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available:    {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Device:        {gpu_name} ({vram_gib:.1f} GiB VRAM)")
        device = torch.device("cuda")
    else:
        print("  WARNING: CUDA is not available! Training will run on CPU.")
        device = torch.device("cpu")

    # 2. Configurations & Paths
    print("\n[2/6] Checking Configurations & Assets...")
    configs = ["configs/base.json", "configs/scenario.json"]
    for cfg_path in configs:
        p = Path(cfg_path)
        if not p.exists():
            print(f"  FAIL: Config file missing: {cfg_path}")
            all_ok = False
            continue
        cfg = load_config(str(p))
        print(f"  Config '{cfg_path}' loaded successfully.")
        
        # Check emulator & ROM
        mesen_exe = Path(cfg.mesen.executable)
        mesen_rom = Path(cfg.mesen.rom)
        bridge_lua = Path(cfg.mesen.lua_script)
        
        if mesen_exe.exists():
            print(f"    Mesen Executable: FOUND ({mesen_exe})")
        else:
            print(f"    FAIL: Mesen Executable NOT found: {mesen_exe}")
            all_ok = False
            
        if mesen_rom.exists():
            print(f"    Tennis ROM:       FOUND ({mesen_rom})")
        else:
            print(f"    FAIL: Tennis ROM NOT found: {mesen_rom}")
            all_ok = False
            
        if bridge_lua.exists():
            print(f"    Bridge Lua:       FOUND ({bridge_lua})")
        else:
            print(f"    FAIL: Bridge Lua script NOT found: {bridge_lua}")
            all_ok = False

    # 3. Lua Bridge Syntax Check
    print("\n[3/6] Checking Mesen Lua Bridge script syntax...")
    try:
        import lupa
        lua = lupa.LuaRuntime()
        bridge_code = Path("mesen/bridge.lua").read_text(encoding="utf-8")
        # Load string as chunk without executing emu-specific calls
        lua.eval(f"function() local f, err = load({bridge_code!r}, '@bridge.lua'); if not f then error(err) end end")()
        print("  Lua Bridge script: Valid Lua 5.1/5.4 chunk compiled with lupa.")
    except Exception as e:
        print(f"  FAIL: Lua Bridge syntax error: {e}")
        all_ok = False

    # 4. Demonstrations & Scenario Manifest
    print("\n[4/6] Checking Demonstration Data & Scenarios...")
    demo_root = "demonstrations"
    try:
        demo_ds = DemonstrationDataset(demo_root, window_length=32)
        demo_summary = summarize_demos(demo_root)
        print(f"  Demonstrations loaded: {len(demo_ds)} points "
              f"({demo_summary['winning_points']} winning, {demo_summary['losing_points']} losing, "
              f"{demo_summary['steps']} total steps).")
    except Exception as e:
        print(f"  FAIL: Could not load demonstrations: {e}")
        all_ok = False

    manifest_path = Path("scenario_states/manifest.jsonl")
    if manifest_path.exists():
        records = load_manifest(manifest_path, base_dir=Path("scenario_states"))
        train_count = sum(1 for r in records if r.split == "train")
        holdout_count = len(records) - train_count
        print(f"  Scenario Manifest: {len(records)} states ({train_count} train, {holdout_count} holdout).")
    else:
        print(f"  WARNING: Scenario manifest not found at {manifest_path}.")

    # 5. Model, DQfD & Segment Replayer Initialisation
    print("\n[5/6] Initializing Neural Network & Segment Replayer...")
    try:
        cfg = load_config("configs/scenario.json")
        base_model = RecurrentActorCritic(
            4, cfg.environment.action_count, cfg.network.channels,
            cfg.network.embedding_dim, cfg.network.gru_hidden, cfg.network.gru_layers,
            input_size=(cfg.mesen.observation_height, cfg.mesen.observation_width)
        ).to(device)

        q_model = RecurrentDuelingQ(base_model).to(device)
        target = RecurrentDuelingQ(RecurrentActorCritic(
            4, cfg.environment.action_count, cfg.network.channels,
            cfg.network.embedding_dim, cfg.network.gru_hidden, cfg.network.gru_layers,
            input_size=(cfg.mesen.observation_height, cfg.mesen.observation_width)
        )).to(device)

        # Dry-run forward pass
        dummy_input = torch.zeros((1, 1, cfg.environment.stack_size, cfg.mesen.observation_height, cfg.mesen.observation_width), device=device)
        h = q_model.initial_hidden(1, device)
        q_out, _ = q_model(dummy_input, h)
        print(f"  Model Forward Pass: OK (Q-values shape: {list(q_out.shape)})")

        # Segment Replayer
        seg_cfg = SegmentReplayConfig(
            seg_len=cfg.ppo.seg_replay_seg_len,
            batch_size=cfg.ppo.seg_replay_batch_size,
            steps_per_update=cfg.ppo.seg_replay_steps_per_update,
            winning_bc_weight=cfg.ppo.seg_replay_winning_bc_weight,
            losing_bc_weight=cfg.ppo.seg_replay_losing_bc_weight,
            outcome_coef=cfg.ppo.seg_replay_outcome_coef,
            learning_rate=cfg.ppo.seg_replay_lr,
        )
        replayer = DemoSegmentReplayer(demo_ds, base_model, device, cfg=seg_cfg)
        print(f"  Segment Replayer: Initialized with {len(replayer._segments)} micro-segments.")
        metrics = replayer.update()
        print(f"  Segment Replayer Dry-Run Update: BC loss={metrics['seg_bc_loss']:.4f}, Outcome loss={metrics['seg_outcome_loss']:.4f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: Model/Replayer initialization error: {e}")
        all_ok = False

    # 6. Checkpoint & Log directories
    print("\n[6/6] Checking Directory Write Permissions...")
    dirs_to_check = [Path("checkpoints_native256x240_frame1"), Path("runs_native256x240_frame1"), Path("checkpoints_hierarchical_native256x240_frame1"), Path("runs_hierarchical_native256x240_frame1"), Path("runs/mesen_ipc")]
    for d in dirs_to_check:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            print(f"  Directory '{d}': Writable.")
        except Exception as e:
            print(f"  FAIL: Directory '{d}' write failed: {e}")
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print(" ALL PRE-TRAINING CHECKS PASSED! SYSTEM IS READY TO TRAIN.")
    else:
        print(" SOME CHECKS FAILED. Please review the output above.")
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
