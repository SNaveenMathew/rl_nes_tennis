from __future__ import annotations

import argparse
import time
import numpy as np

from .config import load_config
from .env import ParallelTennis
from .train import build_workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mesen environment throughput.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    workers = build_workers(cfg)
    env = ParallelTennis(
        workers,
        stack_size=cfg.environment.stack_size,
        max_steps=cfg.environment.episode_max_steps,
    )
    try:
        env.reset()
        actions = np.zeros(env.n, dtype=np.int64)
        start = time.perf_counter()
        steps = 0
        observations = 0
        while time.perf_counter() - start < args.seconds:
            env.step(actions)
            steps += env.n
            observations += env.n
        elapsed = time.perf_counter() - start
        print(f"workers={env.n}")
        print(f"elapsed={elapsed:.2f}s")
        print(f"env_steps={steps:,}")
        print(f"environment_fps={steps/max(elapsed, 1e-9):.2f}")
        print(f"observations={observations:,}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
