"""Train/evaluate hierarchy stages from explicit task/state demonstrations.

Recommended sequence:
  1) foundation: shared visual/recurrent encoder + all leaf heads from labels
  2) leaf --skill TASK: specialize one leaf with the shared encoder frozen
  3) assemble_hierarchy: combine compatible specialist heads
  4) gates: learn root phase gate + phase-conditional skill gates
  5) train --algo hierarchical: online point-level RL fine-tuning

No labels are synthesized from action frequency or point outcome. Missing labels
are masked. Checkpoints contain the complete model state and stage metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import load_config
from .hierarchical import HierarchicalActorCritic
from .hierarchy_schema import EVENT_NAMES, PHASES, TASK_TO_PHASE
from .task_data import TaskAnnotationDataset
from .demonstrations import validate_demo_root_compatibility


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(cfg) -> HierarchicalActorCritic:
    hc = cfg.hierarchical
    return HierarchicalActorCritic(
        observation_channels=cfg.environment.stack_size,
        action_count=cfg.environment.action_count,
        channels=cfg.network.channels,
        embedding_dim=cfg.network.embedding_dim,
        hidden_dim=hc.hidden_dim,
        skills=hc.skills,
        durations=hc.durations,
        input_size=(cfg.mesen.observation_height, cfg.mesen.observation_width),
        button_pulse_durations=hc.button_pulse_durations,
        gru_hidden_dim=cfg.network.gru_hidden,
        gru_layers=cfg.network.gru_layers,
    )


def load_weights(model: HierarchicalActorCritic, path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a model state_dict")
    result = model.load_state_dict(state, strict=False)
    # Stage checkpoints must have the same hierarchy. Partial legacy loads are
    # rejected: silently missing gates or leaf heads makes evaluation misleading.
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"Checkpoint architecture mismatch: missing={result.missing_keys[:8]}, "
            f"unexpected={result.unexpected_keys[:8]}. Start from a v2 hierarchy checkpoint."
        )
    if isinstance(payload, dict) and payload.get("skills") and tuple(payload["skills"]) != model.skills:
        raise ValueError("Checkpoint skill ordering does not match config")
    return payload if isinstance(payload, dict) else {}


def configure_trainable(model: HierarchicalActorCritic, stage: str, skill: str | None) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    trained: list[str] = []
    if stage in ("foundation", "joint"):
        prefixes = ("encoder.", "belief_gru.", "phase_head.", "manager.", "manager_duration_head.",
                    "manager_value_head.", "branch_gates.", "experts.", "ball_position_head.",
                    "player_position_head.", "ball_velocity_head.", "time_to_contact_head.",
                    "time_to_contact_bin_head.", "event_head.")
        for name, p in model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad_(True)
        trained.extend(["shared_encoder", "all_leaf_heads", "gates", "perception_heads"])
    elif stage == "perception":
        prefixes = ("encoder.", "belief_gru.", "ball_position_head.", "player_position_head.",
                    "ball_velocity_head.", "time_to_contact_head.", "time_to_contact_bin_head.", "event_head.")
        for name, p in model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad_(True)
        trained.extend(["shared_encoder", "perception_heads"])
    elif stage == "leaf":
        if not skill or skill not in model.skill_to_id:
            raise ValueError(f"--stage leaf requires --skill from: {', '.join(model.skills)}")
        for name, p in model.experts[skill].named_parameters():
            p.requires_grad_(True)
        trained.append(f"experts.{skill}")
    elif stage == "gates":
        prefixes = ("phase_head.", "manager.", "manager_duration_head.", "manager_value_head.", "branch_gates.")
        for name, p in model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad_(True)
        trained.extend(["root_phase_gate", "conditional_skill_gates", "duration_gate"])
    else:
        raise ValueError(f"Unknown stage {stage}")
    return trained


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target).all(dim=-1)
    if valid.any():
        return nn.functional.smooth_l1_loss(pred[valid], target[valid])
    return pred.sum() * 0.0


def perception_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), device=device)
    metrics: dict[str, float] = {}
    terms = 0
    for key, output_key in (("state_ball_xy", "ball_xy"), ("state_player_xy", "player_xy"),
                            ("state_ball_velocity_xy", "ball_velocity_xy")):
        target = batch.get(key)
        if target is not None:
            target = target.to(device).float()
            term = _masked_smooth_l1(out[output_key], target)
            if torch.isfinite(target).any():
                loss = loss + term
                terms += 1
                metrics[f"{key}_loss"] = float(term.detach())
    target = batch.get("state_time_to_contact")
    if target is not None:
        target = target.to(device).float().reshape(-1)
        valid = torch.isfinite(target) & (target >= 0)
        if valid.any():
            term = nn.functional.smooth_l1_loss(out["time_to_contact"][valid], target[valid])
            loss, terms = loss + term, terms + 1
            metrics["time_to_contact_loss"] = float(term.detach())
    target = batch.get("state_time_to_contact_bin")
    if target is not None:
        target = target.to(device).long().reshape(-1)
        valid = (target >= 0) & (target < 3)
        if valid.any():
            term = nn.functional.cross_entropy(out["time_to_contact_logits"][valid], target[valid])
            loss, terms = loss + term, terms + 1
            metrics["time_to_contact_bin_loss"] = float(term.detach())
    target = batch.get("state_events")
    if target is not None:
        target = target.to(device).float()
        pred = out["event_logits"]
        if target.shape[-1] < pred.shape[-1]:
            target = torch.cat([target, torch.full((len(target), pred.shape[-1]-target.shape[-1]), -1., device=device)], -1)
        valid = (target >= 0) & (target <= 1)
        if valid.any():
            values = nn.functional.binary_cross_entropy_with_logits(pred, target.clamp(0, 1), reduction="none")
            term = (values * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            loss, terms = loss + term, terms + 1
            metrics["event_loss"] = float(term.detach())
    return loss, metrics


def _to_device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def stage_losses(model: HierarchicalActorCritic, batch: dict[str, torch.Tensor], stage: str,
                 skill: str | None, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    batch = _to_device_batch(batch, device)
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    zero = next(model.parameters()).sum() * 0.0
    # Leaf behavior cloning from explicitly labeled actions only.
    d_obs, d_skill, d_action = batch["demo_obs"], batch["demo_skill"], batch["demo_action"]
    if stage in ("foundation", "leaf", "joint") and d_obs.shape[0] > 0:
        valid = d_skill >= 0
        if skill is not None:
            valid &= torch.tensor([model.skills[int(i)] == skill for i in d_skill.tolist()], device=device)
        if valid.any():
            out = model(d_obs[valid])
            rows = torch.arange(int(valid.sum()), device=device)
            target_skill = d_skill[valid]
            action_logits = out["expert_logits"][rows, target_skill]
            losses["leaf_action"] = nn.functional.cross_entropy(action_logits, d_action[valid])
            metrics["leaf_action_loss"] = float(losses["leaf_action"].detach())
            pulse_ids = batch["demo_pulse_duration_ids"][valid]
            pulse_valid = (pulse_ids >= 0) & (d_action[valid].remainder(3) != 0)
            if pulse_valid.any():
                pulse_logits = out["expert_pulse_logits"][rows[pulse_valid], target_skill[pulse_valid]]
                losses["pulse"] = nn.functional.cross_entropy(pulse_logits, pulse_ids[pulse_valid])
                metrics["pulse_loss"] = float(losses["pulse"].detach())

    # The gate is supervised only by annotation labels; a broad legacy task
    # trains phase but doesn't invent an exact leaf target.
    m_obs = batch["demo_manager_obs"]
    if stage in ("foundation", "gates", "joint") and m_obs.shape[0] > 0:
        out = model(m_obs)
        phases = batch["demo_manager_phase"]
        phase_valid = phases >= 0
        if phase_valid.any():
            losses["phase"] = nn.functional.cross_entropy(out["phase_logits"][phase_valid], phases[phase_valid])
            metrics["phase_loss"] = float(losses["phase"].detach())
        leaves = batch["demo_manager_skill"]
        leaf_valid = (leaves >= 0) & phase_valid
        if leaf_valid.any():
            # Train each conditional gate against leaf labels from its own
            # ground-truth phase. A flat global 13-way CE confounds phase and
            # within-phase discrimination and is ill-conditioned for branches
            # of different widths.
            weighted_terms: list[torch.Tensor] = []
            sample_count = 0
            for phase_name in PHASES:
                phase_id = PHASES.index(phase_name)
                ids = model.phase_task_ids[phase_name]
                mask = leaf_valid & (phases == phase_id)
                if not ids or not mask.any():
                    continue
                global_to_local = {global_id: local_id for local_id, global_id in enumerate(ids)}
                local_targets = torch.tensor(
                    [global_to_local[int(x)] for x in leaves[mask].tolist()],
                    dtype=torch.long, device=device,
                )
                local_logits = out["branch_logits"][phase_name][mask]
                term = nn.functional.cross_entropy(local_logits, local_targets, reduction="sum")
                weighted_terms.append(term)
                sample_count += int(mask.sum())
            if sample_count:
                losses["gate_leaf"] = torch.stack(weighted_terms).sum() / sample_count
                metrics["gate_leaf_loss"] = float(losses["gate_leaf"].detach())
        durations = batch["demo_manager_duration"]
        duration_valid = phase_valid & (durations >= 0)
        if duration_valid.any():
            losses["duration"] = nn.functional.cross_entropy(out["manager_duration_logits"][duration_valid], durations[duration_valid])
            metrics["duration_loss"] = float(losses["duration"].detach())

    s_obs = batch.get("state_obs")
    if stage in ("foundation", "perception", "joint") and s_obs is not None and s_obs.shape[0] > 0:
        out = model(s_obs)
        losses["perception"], per_metrics = perception_loss(out, batch, device)
        metrics.update(per_metrics)
        metrics["perception_loss"] = float(losses["perception"].detach())

    if not losses:
        if stage == "perception":
            raise ValueError("No usable state annotations in this split. Add state_annotations.jsonl first.")
        if stage == "leaf":
            raise ValueError(f"No exact leaf action labels for {skill!r} in this split. Add task_annotations.jsonl.")
        raise ValueError(f"No applicable labels were found for stage={stage!r} in this data split.")

    # Foundation puts most weight on real action targets and lets explicitly
    # labeled state prediction help the shared representation. No pseudo-labels.
    weights = {"leaf_action": 1.0, "pulse": 0.5, "phase": 0.5, "gate_leaf": 0.5,
               "duration": 0.2, "perception": 0.5}
    if stage == "gates":
        weights = {k: (1.0 if k in ("phase", "gate_leaf", "duration") else 0.0) for k in weights}
    elif stage == "leaf":
        weights = {k: (1.0 if k in ("leaf_action", "pulse") else 0.0) for k in weights}
    elif stage == "perception":
        weights = {k: (1.0 if k == "perception" else 0.0) for k in weights}
    total = zero
    for key, value in losses.items():
        total = total + weights.get(key, 0.0) * value
    metrics["loss"] = float(total.detach())
    return total, metrics


@torch.no_grad()
def evaluate_stage(model: HierarchicalActorCritic, data: TaskAnnotationDataset, stage: str,
                   skill: str | None, device: torch.device, chunk: int = 128) -> dict[str, float]:
    model.eval()
    stats: dict[str, float] = {}
    if stage in ("foundation", "leaf", "joint"):
        correct = total = pulse_correct = pulse_total = 0
        obs_rows: list[np.ndarray] = []
        action_rows: list[int] = []
        skill_rows: list[int] = []
        pulse_rows: list[int] = []
        for seg in data.segments:
            if seg.leaf_task is None or (skill is not None and seg.leaf_task != skill):
                continue
            sk = data.skills.index(seg.leaf_task)
            for i in range(seg.length):
                obs_rows.append(seg.observations[i]); action_rows.append(int(seg.actions[i])); skill_rows.append(sk)
                raw = -1 if seg.pulse_frames is None else int(seg.pulse_frames[i])
                pulse_rows.append(min(range(len(model.button_pulse_durations)), key=lambda j: abs(model.button_pulse_durations[j]-raw)) if raw > 0 and int(seg.actions[i]) % 3 else -1)
        for start in range(0, len(obs_rows), chunk):
            x = torch.from_numpy(np.stack(obs_rows[start:start+chunk])).to(device)
            out = model(x)
            skills = torch.tensor(skill_rows[start:start+chunk], dtype=torch.long, device=device)
            actions = torch.tensor(action_rows[start:start+chunk], dtype=torch.long, device=device)
            rows = torch.arange(len(x), device=device)
            pred = out["expert_logits"][rows, skills].argmax(-1)
            correct += int((pred == actions).sum()); total += len(x)
            pids = torch.tensor(pulse_rows[start:start+chunk], dtype=torch.long, device=device)
            valid = pids >= 0
            if valid.any():
                p = out["expert_pulse_logits"][rows[valid], skills[valid]].argmax(-1)
                pulse_correct += int((p == pids[valid]).sum()); pulse_total += int(valid.sum())
        stats["val_leaf_action_accuracy"] = correct / total if total else float("nan")
        stats["val_leaf_action_examples"] = float(total)
        stats["val_pulse_accuracy"] = pulse_correct / pulse_total if pulse_total else float("nan")
        stats["val_pulse_examples"] = float(pulse_total)
    if stage in ("foundation", "gates", "joint"):
        correct_phase = total_phase = correct_leaf = total_leaf = correct_route = total_route = 0
        rows = [seg for seg in data.segments if seg.phase and (skill is None or seg.leaf_task in (None, skill))]
        for start in range(0, len(rows), chunk):
            group = rows[start:start+chunk]
            x = torch.from_numpy(np.stack([seg.observations[0] for seg in group])).to(device)
            out = model(x)
            true_phase_ids = torch.tensor([PHASES.index(seg.phase) for seg in group], device=device)
            predicted_phase_ids = out["phase_logits"].argmax(-1)
            correct_phase += int((predicted_phase_ids == true_phase_ids).sum()); total_phase += len(group)
            valid_local = [i for i, seg in enumerate(group) if seg.leaf_task is not None]
            for i in valid_local:
                target_skill = data.skills.index(group[i].leaf_task)
                true_phase = group[i].phase
                local_targets = model.phase_task_ids[true_phase]
                target_local = local_targets.index(target_skill)
                branch_pred = int(out["branch_logits"][true_phase][i].argmax(-1))
                correct_leaf += int(branch_pred == target_local); total_leaf += 1
                predicted_phase = PHASES[int(predicted_phase_ids[i])]
                predicted_local = int(out["branch_logits"][predicted_phase][i].argmax(-1))
                predicted_global = model.phase_task_ids[predicted_phase][predicted_local]
                correct_route += int(predicted_global == target_skill); total_route += 1
        stats["val_phase_accuracy"] = correct_phase / total_phase if total_phase else float("nan")
        stats["val_phase_examples"] = float(total_phase)
        stats["val_gate_leaf_accuracy"] = correct_leaf / total_leaf if total_leaf else float("nan")
        stats["val_gate_leaf_examples"] = float(total_leaf)
        stats["val_manager_skill_accuracy"] = correct_route / total_route if total_route else float("nan")
        stats["val_manager_skill_examples"] = float(total_route)
    if stage in ("foundation", "perception", "joint") and data.state_annotations:
        ball_abs = []; player_abs = []; event_correct = event_total = 0
        for start in range(0, len(data.state_annotations), chunk):
            group = data.state_annotations[start:start+chunk]
            x = torch.from_numpy(np.stack([v.observation for v in group])).to(device)
            out = model(x)
            for key, field in (("ball_xy", "ball_xy"), ("player_xy", "player_xy")):
                targets = np.array([getattr(v, field) if getattr(v, field) is not None else [np.nan, np.nan] for v in group], dtype=np.float32)
                t = torch.from_numpy(targets).to(device)
                valid = torch.isfinite(t).all(-1)
                if valid.any():
                    err = (out[key][valid] - t[valid]).abs().mean(-1)
                    (ball_abs if field == "ball_xy" else player_abs).extend(err.cpu().tolist())
            target_events = np.asarray([v.events for v in group], dtype=np.float32)
            t = torch.from_numpy(target_events).to(device)
            valid = (t >= 0) & (t <= 1)
            if valid.any():
                pred = (torch.sigmoid(out["event_logits"]) >= 0.5).float()
                event_correct += int((pred[valid] == t[valid]).sum()); event_total += int(valid.sum())
        stats["val_ball_mae"] = float(np.mean(ball_abs)) if ball_abs else float("nan")
        stats["val_player_mae"] = float(np.mean(player_abs)) if player_abs else float("nan")
        stats["val_event_accuracy"] = event_correct / event_total if event_total else float("nan")
        stats["val_state_examples"] = float(len(data.state_annotations))
    return stats


def _save(path: Path, model: HierarchicalActorCritic, cfg, stage: str, skill: str | None,
          update: int, metrics: dict[str, float], source: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "tennis_rl_hierarchical_v2", "model": model.state_dict(),
        "skills": list(model.skills), "durations": list(model.durations),
        "button_pulse_durations": list(model.button_pulse_durations),
        "stage": stage, "skill": skill, "updates": update, "metrics": metrics,
        "init_from": str(source or ""),
        "config": {"channels": cfg.network.channels, "embedding_dim": cfg.network.embedding_dim,
                   "hidden_dim": cfg.hierarchical.hidden_dim, "gru_hidden": cfg.network.gru_hidden,
                   "gru_layers": cfg.network.gru_layers,
                   "mesen": {"observation_width": cfg.mesen.observation_width,
                             "observation_height": cfg.mesen.observation_height,
                             "frame_skip": cfg.mesen.frame_skip}},
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario.json")
    parser.add_argument("--stage", required=True, choices=("foundation", "perception", "leaf", "gates", "joint"))
    parser.add_argument("--skill", choices=sorted(TASK_TO_PHASE), help="Required for --stage leaf; optional task filter for evaluation")
    parser.add_argument("--demo-root", default=None)
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--init-from", default="", help="Load weights only, not optimizer/checkpoint counters")
    parser.add_argument("--output", default="")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("updates, batch-size, and eval-every must be positive")
    if args.stage == "leaf" and not args.skill:
        parser.error("--stage leaf requires --skill")

    cfg = load_config(args.config)
    try:
        validate_demo_root_compatibility(
            args.demo_root or cfg.ppo.demo_root,
            observation_height=cfg.mesen.observation_height,
            observation_width=cfg.mesen.observation_width,
            frame_skip=cfg.mesen.frame_skip,
            stack_size=cfg.environment.stack_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.device:
        cfg.device = args.device
    seed_all(args.seed)
    device = torch.device(cfg.device)
    model = make_model(cfg).to(device)
    if args.init_from:
        load_weights(model, args.init_from, device)
        print(f"Loaded weights (not optimizer state): {args.init_from}")
    elif args.stage == "leaf":
        parser.error("Independent leaf training requires --init-from foundation.pt so the shared encoder stays compatible")
    trainable = configure_trainable(model, args.stage, args.skill)
    demo_root = Path(args.demo_root or cfg.ppo.demo_root)
    train_data = TaskAnnotationDataset(demo_root, model.skills, model.durations,
        button_pulse_durations=model.button_pulse_durations, split=args.split,
        holdout_fraction=args.holdout_fraction)
    holdout_data = TaskAnnotationDataset(demo_root, model.skills, model.durations,
        button_pulse_durations=model.button_pulse_durations, split="holdout",
        holdout_fraction=args.holdout_fraction)
    if args.stage == "leaf" and not any(s.leaf_task == args.skill for s in train_data.segments):
        raise SystemExit(f"No exact '{args.skill}' training annotations in {demo_root} split={args.split}. See `python -m tennis_rl.dataset stats`.")
    if args.stage == "perception" and not train_data.state_annotations:
        raise SystemExit(f"No state annotations in {demo_root} split={args.split}; add state_annotations.jsonl.")
    print(json.dumps({"stage": args.stage, "skill": args.skill, "trainable": trainable,
        "train_data": train_data.summary(), "holdout_data": holdout_data.summary(),
        "device": str(device)}, indent=2))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("No parameters selected for training")
    optimizer = torch.optim.AdamW(params, lr=float(cfg.hierarchical.learning_rate), weight_decay=1e-5)
    output = Path(args.output or (Path(cfg.hierarchical.save_dir) / "stages" / f"{args.stage}{'_' + args.skill if args.skill else ''}.pt"))
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".csv")
    rng = np.random.default_rng(args.seed + 1201)
    start = time.perf_counter()
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["update", "train_loss", "leaf_action_loss", "pulse_loss", "phase_loss", "gate_leaf_loss", "duration_loss", "perception_loss", "val_leaf_action_accuracy", "val_phase_accuracy", "val_gate_leaf_accuracy", "val_manager_skill_accuracy", "val_ball_mae", "val_player_mae", "val_event_accuracy"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        last_metrics: dict[str, float] = {}
        for update in range(1, args.updates + 1):
            model.train()
            batch = train_data.sample_training_batch(args.batch_size, rng,
                selected_skills=({args.skill} if args.stage == "leaf" else None))
            if batch is None:
                raise SystemExit("No usable training annotations in this demonstration root/split")
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = stage_losses(model, batch, args.stage, args.skill, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update}: {metrics}")
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            last_metrics = dict(metrics)
            row: dict[str, Any] = {"update": update, "train_loss": metrics["loss"]}
            row.update(metrics)
            if update % args.eval_every == 0 or update == args.updates:
                val = evaluate_stage(model, holdout_data, args.stage, args.skill, device)
                row.update(val)
                elapsed = max(time.perf_counter() - start, 1e-6)
                print(f"stage={args.stage} skill={args.skill or '-'} update={update}/{args.updates} "
                      f"loss={metrics['loss']:.5f} elapsed={elapsed:.1f}s " + " ".join(f"{k}={v:.3f}" for k, v in val.items() if np.isfinite(v)), flush=True)
                last_metrics.update(val)
            writer.writerow({key: row.get(key, "") for key in fields})
            fh.flush()
    _save(output, model, cfg, args.stage, args.skill, args.updates, last_metrics, args.init_from or None)
    print(f"Saved stage checkpoint: {output.resolve()}\nMetrics CSV: {log_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
