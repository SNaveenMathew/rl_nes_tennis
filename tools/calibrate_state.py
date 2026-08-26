from __future__ import annotations

import argparse

from tennis_rl.config import load_config
from tennis_rl.mesen_client import MesenWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a Mesen Tennis starting savestate.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--ready", action="store_true", help="Create the state immediately after bridge startup.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    worker = MesenWorker(
        cfg.mesen.executable,
        cfg.mesen.rom,
        cfg.mesen.lua_script,
        cfg.mesen.host,
        cfg.mesen.port_start,
        cfg.mesen.observation_width,
        cfg.mesen.observation_height,
        cfg.mesen.frame_skip,
        cfg.mesen.startup_timeout_s,
        cfg.mesen.socket_timeout_s,
    )
    try:
        print("Mesen is running. Reach the exact desired Tennis serve/return starting point in the emulator.")
        input("Press Enter to request an in-memory savestate... ")
        worker.save_state()
        # State acknowledgement can interleave with observations; this helper
        # simply leaves the running emulator alive long enough for the callback.
        print("Savestate request sent. Keep this process attached while validating your setup.")
        input("Press Enter to close the calibration worker... ")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
