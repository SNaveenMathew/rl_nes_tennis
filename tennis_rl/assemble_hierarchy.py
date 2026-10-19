"""Assemble independently trained leaf heads against one shared foundation encoder.

Example:
  python -m tennis_rl.assemble_hierarchy --config configs/scenario.json \
    --base checkpoints_hierarchical/stages/foundation.pt \
    --leaf serve_timing=checkpoints_hierarchical/stages/leaf_serve_timing.pt \
    --leaf stroke_timing=checkpoints_hierarchical/stages/leaf_stroke_timing.pt \
    --output checkpoints_hierarchical/stages/leaves_assembled.pt

Each leaf checkpoint must have been trained from the same base/foundation so its
frozen encoder and recurrent state are identical. Only that named expert subtree
is copied into the base checkpoint; gates are trained afterward.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def load(path: Path, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(obj, dict) or obj.get("format") != "tennis_rl_hierarchical_v2" or "model" not in obj:
        raise ValueError(f"{path} is not a tennis_rl_hierarchical_v2 stage checkpoint")
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--leaf", action="append", default=[], metavar="TASK=CHECKPOINT")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    device = torch.device(a.device)
    base_path = Path(a.base).resolve()
    base = load(base_path, device)
    state = dict(base["model"])
    skills = tuple(base["skills"])
    loaded_tasks = []
    for spec in a.leaf:
        if "=" not in spec:
            p.error(f"--leaf must be TASK=CHECKPOINT, got {spec!r}")
        task, path_text = spec.split("=", 1)
        if task not in skills:
            raise ValueError(f"Unknown task {task!r}; configured skills: {skills}")
        path = Path(path_text).resolve()
        payload = load(path, device)
        if tuple(payload["skills"]) != skills:
            raise ValueError(f"Skill ordering differs in {path}")
        if payload.get("skill") != task or payload.get("stage") != "leaf":
            raise ValueError(f"{path} is not a dedicated leaf checkpoint for {task!r}")
        if tuple(payload.get("button_pulse_durations", ())) != tuple(base.get("button_pulse_durations", ())):
            raise ValueError(f"Button pulse schema mismatch for {path}")
        prefix = f"experts.{task}."
        target_keys = [key for key in state if key.startswith(prefix)]
        source_keys = [key for key in payload["model"] if key.startswith(prefix)]
        if not target_keys or set(target_keys) != set(source_keys):
            raise ValueError(f"Expert parameter schema mismatch for {task!r}")
        # Critical invariant: all stage encoders must start from the same base;
        # the per-leaf command freezes it, so mismatches signal accidental branch drift.
        for key in state:
            if key.startswith("encoder.") or key.startswith("belief_gru."):
                if not torch.equal(state[key].cpu(), payload["model"][key].cpu()):
                    raise ValueError(f"Shared encoder changed in {path}. Train leaf heads with default encoder freeze.")
        for key in target_keys:
            state[key] = payload["model"][key]
        loaded_tasks.append(task)
    base["model"] = state
    base["stage"] = "assembled_leaves"
    base["assembled_leaf_tasks"] = loaded_tasks
    base["optimizer"] = None
    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(f"Assembled {len(loaded_tasks)} leaf heads into {out}\nTasks: {', '.join(loaded_tasks) if loaded_tasks else '(none copied)'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
