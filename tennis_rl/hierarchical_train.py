"""Training/evaluation loop for the hierarchical manager + specialist policies."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.distributions import Categorical

from .hierarchical import HierarchicalActorCritic, HierarchicalController, HierarchicalPPOTrainer
from .scenario_sampler import SamplerConfig, ScenarioLibrary, ScenarioSampler
from .scenarios import load_manifest
from .env import ParallelTennis
from .task_data import TaskAnnotationDataset


def _verified_manager_targets(assignment: Any, model: HierarchicalActorCritic, enabled: bool) -> tuple[int, int]:
    """Return (exact leaf target, broad phase target) from trusted metadata only."""
    if not enabled or assignment is None:
        return -1, -1
    if getattr(assignment, "kind", None) not in ("scenario", "probe"):
        return -1, -1
    if getattr(assignment, "phase", None) != "start":
        return -1, -1
    role = str(getattr(assignment, "cell_id", "")).split("|", 1)[0]
    phase = {"serve": "serve", "return": "receive"}.get(role)
    if phase is None:
        return -1, -1
    return -1, model.phase_to_id.get(phase, -1)


def collect_rollout(
    model: HierarchicalActorCritic,
    env: ParallelTennis,
    observations: np.ndarray,
    cfg,
    device: torch.device,
    belief_hidden: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], np.ndarray, dict[str, float]]:
    """Collect option-level semi-Markov and leaf-level PPO data.

    The GRU state is persistent across observation boundaries and is reset only
    when the environment truly ends/restarts. Returned ``data['belief_hidden_out']``
    is the state to pass to the next rollout; it is intentionally removed before
    trainer.update() because it is collector state, not a training target.
    """
    hc = cfg.hierarchical
    t_steps, n_envs = int(hc.rollout_steps), int(env.n)
    if belief_hidden is None:
        belief_hidden = model.initial_belief(n_envs, device)
    else:
        belief_hidden = belief_hidden.to(device)
    if max(model.button_pulse_durations) > int(cfg.mesen.frame_skip):
        raise ValueError(
            f"hierarchical.button_pulse_durations={model.button_pulse_durations} must be <= "
            f"mesen.frame_skip={cfg.mesen.frame_skip}; pulses cannot exceed one observed action interval"
        )

    skills_buf = np.zeros((t_steps, n_envs), dtype=np.int64)
    expert_values_buf = np.zeros((t_steps, n_envs), dtype=np.float32)
    rewards_buf = np.zeros((t_steps, n_envs), dtype=np.float32)
    dones_buf = np.zeros((t_steps, n_envs), dtype=np.bool_)
    truncs_buf = np.zeros((t_steps, n_envs), dtype=np.bool_)
    next_values_buf = np.zeros((t_steps, n_envs), dtype=np.float32)

    manager_records: list[list[dict[str, Any]]] = [[] for _ in range(n_envs)]
    contexts: list[dict[str, Any] | None] = [None] * n_envs
    active_skills = np.full(n_envs, -1, dtype=np.int64)
    remaining = np.zeros(n_envs, dtype=np.int64)
    expert_obs_buf: list[np.ndarray] = []
    expert_skill_list: list[int] = []
    expert_action_list: list[int] = []
    expert_pulse_list: list[int] = []
    expert_logp_list: list[float] = []
    expert_value_list: list[float] = []
    expert_step_worker: list[int] = []
    expert_step_index: list[int] = []
    expert_hidden_list: list[torch.Tensor] = []
    stats = Counter()
    model.eval()

    for t in range(t_steps):
        obs_t = torch.from_numpy(observations).to(device, non_blocking=True)
        hidden_before = belief_hidden.detach().clone()
        with torch.no_grad():
            output = model(obs_t, belief_hidden)
            hidden_after = output["belief_hidden_out"].detach()
            skill_dist = Categorical(logits=output["manager_skill_logits"])
            duration_dist = Categorical(logits=output["manager_duration_logits"])
            sampled_skills = skill_dist.sample()
            sampled_durations = duration_dist.sample()
            sampled_skill_logp = skill_dist.log_prob(sampled_skills)
            sampled_duration_logp = duration_dist.log_prob(sampled_durations)

            # A manager action is sampled only at a learned option boundary.
            for i in range(n_envs):
                if contexts[i] is None or remaining[i] <= 0:
                    skill_id = int(sampled_skills[i].item())
                    duration_id = int(sampled_durations[i].item())
                    manager_logp = float((sampled_skill_logp[i] + sampled_duration_logp[i]).item())
                    target_leaf, target_phase = _verified_manager_targets(
                        env.assignments[i], model, bool(hc.use_verified_start_role_labels)
                    )
                    contexts[i] = {
                        "obs": observations[i].copy(), "skill": skill_id,
                        "duration": duration_id, "logp": manager_logp,
                        "value": float(output["manager_values"][i].item()),
                        "hidden": hidden_before[:, i, :].detach().cpu(),
                        "return": 0.0, "elapsed": 0,
                        "target": target_leaf, "target_phase": target_phase,
                    }
                    active_skills[i] = skill_id
                    remaining[i] = int(model.durations[duration_id])

            selected_skills = torch.as_tensor(active_skills, dtype=torch.long, device=device)
            rows = torch.arange(n_envs, device=device)
            selected_logits = output["expert_logits"][rows, selected_skills]
            expert_dist = Categorical(logits=selected_logits)
            action_tensor = expert_dist.sample()
            expert_logp = expert_dist.log_prob(action_tensor)
            selected_pulse_logits = output["expert_pulse_logits"][rows, selected_skills]
            pulse_dist = Categorical(logits=selected_pulse_logits)
            pulse_ids = pulse_dist.sample()
            pulse_logp = pulse_dist.log_prob(pulse_ids)
            has_shot = (action_tensor.remainder(3) != 0).float()
            combined_expert_logp = expert_logp + pulse_logp * has_shot
            expert_value = output["expert_values"][rows, selected_skills]

        action_np = action_tensor.cpu().numpy().astype(np.int64, copy=False)
        pulse_id_np = pulse_ids.cpu().numpy().astype(np.int64, copy=False)
        pulse_frames_np = np.asarray(
            [model.button_pulse_durations[int(index)] for index in pulse_id_np], dtype=np.int64
        )
        batch = env.step(action_np, pulse_frames=pulse_frames_np)
        next_obs = batch.observations
        next_obs_t = torch.from_numpy(next_obs).to(device, non_blocking=True)
        # Evaluate terminal/truncated next state using the hidden state after
        # the current observation. This call does not advance the persistent
        # belief; the next collection iteration will process the same next state.
        with torch.no_grad():
            next_output = model(next_obs_t, hidden_after)
        done_arr = np.asarray(batch.dones, dtype=np.bool_)
        trunc_arr = np.asarray(batch.truncateds if batch.truncateds is not None else [False] * n_envs, dtype=np.bool_)
        term_arr = np.asarray([
            bool(info.get("terminated", bool(done_arr[i] and not trunc_arr[i])))
            for i, info in enumerate(batch.infos)
        ], dtype=np.bool_)

        for i in range(n_envs):
            action = int(action_np[i])
            skill_id = int(active_skills[i])
            value = float(expert_value[i].item())
            logp = float(combined_expert_logp[i].item())
            reward = float(batch.rewards[i])
            next_value = float(next_output["expert_values"][i, skill_id].item())
            expert_obs_buf.append(observations[i].copy())
            expert_skill_list.append(skill_id)
            expert_action_list.append(action)
            expert_pulse_list.append(int(pulse_id_np[i]))
            expert_logp_list.append(logp)
            expert_value_list.append(value)
            expert_hidden_list.append(hidden_before[:, i, :].detach().cpu())
            expert_step_worker.append(i)
            expert_step_index.append(t)

            skills_buf[t, i] = skill_id
            expert_values_buf[t, i] = value
            rewards_buf[t, i] = reward
            dones_buf[t, i] = bool(done_arr[i])
            truncs_buf[t, i] = bool(trunc_arr[i])
            next_values_buf[t, i] = next_value

            context = contexts[i]
            if context is not None:
                context["return"] += (float(hc.gamma) ** int(context["elapsed"])) * reward
                context["elapsed"] += 1
            remaining[i] -= 1

            info = batch.infos[i]
            if info.get("service_fault"):
                stats["service_faults"] += 1
            if info.get("rally_survival"):
                stats["rally_survivals"] += 1
            if info.get("bridge_restart"):
                stats["bridge_restarts"] += 1
            if done_arr[i]:
                stats["episodes"] += 1
                result = int(info.get("point_result", 0))
                if result > 0:
                    stats["point_wins"] += 1
                elif result < 0:
                    stats["point_losses"] += 1
                else:
                    stats["unknown_point_results"] += 1

            option_finished = remaining[i] <= 0 or bool(done_arr[i])
            if option_finished and context is not None:
                manager_next_value = float(next_output["manager_values"][i].item())
                if term_arr[i]:
                    manager_next_value = 0.0
                manager_records[i].append({
                    **context,
                    "next_value": manager_next_value,
                    "terminated": bool(term_arr[i]),
                    "truncated": bool(trunc_arr[i]),
                    "duration_steps": int(context["elapsed"]),
                })
                contexts[i] = None
                active_skills[i] = -1
                remaining[i] = 0

        # Recurrent memory advances exactly once per current observation. At a
        # completed episode, clear memory before the reset observation next time.
        belief_hidden = hidden_after.clone()
        for i in np.flatnonzero(done_arr):
            belief_hidden[:, int(i), :] = 0.0
        observations = next_obs.copy()
        for i in np.flatnonzero(done_arr):
            observations[int(i)] = env.current_observation(int(i))

    # Rollout boundary is a truncation: bootstrap the next state but keep the
    # persistent hidden state from the last state actually acted upon.
    final_obs_t = torch.from_numpy(observations).to(device, non_blocking=True)
    with torch.no_grad():
        final_output = model(final_obs_t, belief_hidden)
    for i, context in enumerate(contexts):
        if context is None:
            continue
        manager_records[i].append({
            **context,
            "next_value": float(final_output["manager_values"][i].item()),
            "terminated": False,
            "truncated": True,
            "duration_steps": int(context["elapsed"]),
        })

    # Semi-Markov GAE over each worker's ordered manager decisions.
    manager_advantages: list[float] = []
    manager_returns: list[float] = []
    manager_ordered_records: list[dict[str, Any]] = []
    gamma, lam = float(hc.gamma), float(hc.gae_lambda)
    for records in manager_records:
        adv = 0.0
        adv_by_record = [0.0] * len(records)
        returns_by_record = [0.0] * len(records)
        for j in reversed(range(len(records))):
            rec = records[j]
            duration = max(1, int(rec["duration_steps"]))
            discount = gamma ** duration
            terminal = bool(rec["terminated"])
            stop_trace = terminal or bool(rec["truncated"])
            delta = float(rec["return"]) + discount * float(rec["next_value"]) * (0.0 if terminal else 1.0) - float(rec["value"])
            adv = delta + ((gamma * lam) ** duration) * (0.0 if stop_trace else adv)
            adv_by_record[j] = adv
            returns_by_record[j] = adv + float(rec["value"])
        for rec, advantage, target_return in zip(records, adv_by_record, returns_by_record):
            manager_ordered_records.append(rec)
            manager_advantages.append(float(advantage))
            manager_returns.append(float(target_return))

    # Per-expert GAE does not leak across expert switches. Truncated transitions
    # bootstrap from the final observed state; terminated transitions do not.
    t_steps = int(hc.rollout_steps)
    expert_adv_by_step = np.zeros((t_steps, n_envs), dtype=np.float32)
    expert_ret_by_step = np.zeros((t_steps, n_envs), dtype=np.float32)
    adv_next = np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(t_steps)):
        for i in range(n_envs):
            terminal = bool(dones_buf[t, i] and not truncs_buf[t, i])
            delta = rewards_buf[t, i] + gamma * next_values_buf[t, i] * (0.0 if terminal else 1.0) - expert_values_buf[t, i]
            same_skill_next = (t + 1 < t_steps and not dones_buf[t, i] and skills_buf[t + 1, i] == skills_buf[t, i])
            trace = gamma * lam * float(adv_next[i]) if same_skill_next else 0.0
            expert_adv_by_step[t, i] = delta + trace
            expert_ret_by_step[t, i] = expert_adv_by_step[t, i] + expert_values_buf[t, i]
        adv_next = expert_adv_by_step[t].copy()

    flat_expert_adv = [float(expert_adv_by_step[t, i]) for t, i in zip(expert_step_index, expert_step_worker)]
    flat_expert_returns = [float(expert_ret_by_step[t, i]) for t, i in zip(expert_step_index, expert_step_worker)]

    def obs_tensor(rows: list[np.ndarray]) -> torch.Tensor:
        if rows:
            return torch.from_numpy(np.stack(rows).astype(np.uint8, copy=False))
        return torch.empty((0, *observations.shape[1:]), dtype=torch.uint8)

    def hidden_tensor(rows: list[torch.Tensor]) -> torch.Tensor:
        if rows:
            return torch.stack(rows, dim=0).float()
        return torch.empty((0, model.gru_layers, model.belief_hidden_dim), dtype=torch.float32)

    data: dict[str, torch.Tensor] = {
        "manager_obs": obs_tensor([r["obs"] for r in manager_ordered_records]),
        "manager_skill": torch.tensor([r["skill"] for r in manager_ordered_records], dtype=torch.long),
        "manager_duration": torch.tensor([r["duration"] for r in manager_ordered_records], dtype=torch.long),
        "manager_old_logp": torch.tensor([r["logp"] for r in manager_ordered_records], dtype=torch.float32),
        "manager_values_old": torch.tensor([r["value"] for r in manager_ordered_records], dtype=torch.float32),
        "manager_advantages": torch.tensor(manager_advantages, dtype=torch.float32),
        "manager_returns": torch.tensor(manager_returns, dtype=torch.float32),
        "manager_target_skill": torch.tensor([r["target"] for r in manager_ordered_records], dtype=torch.long),
        "manager_target_phase": torch.tensor([r["target_phase"] for r in manager_ordered_records], dtype=torch.long),
        "manager_hidden": hidden_tensor([r["hidden"] for r in manager_ordered_records]),
        "expert_obs": obs_tensor(expert_obs_buf),
        "expert_skill": torch.tensor(expert_skill_list, dtype=torch.long),
        "expert_action": torch.tensor(expert_action_list, dtype=torch.long),
        "expert_pulse_duration_ids": torch.tensor(expert_pulse_list, dtype=torch.long),
        "expert_old_logp": torch.tensor(expert_logp_list, dtype=torch.float32),
        "expert_advantages": torch.tensor(flat_expert_adv, dtype=torch.float32),
        "expert_returns": torch.tensor(flat_expert_returns, dtype=torch.float32),
        "expert_hidden": hidden_tensor(expert_hidden_list),
        "belief_hidden_out": belief_hidden.detach().cpu(),
    }
    stats_out = {k: float(v) for k, v in stats.items()}
    stats_out["manager_decisions"] = float(len(manager_ordered_records))
    stats_out["expert_steps"] = float(len(expert_obs_buf))
    return data, observations, stats_out

def _scenario_sampler(cfg, seed: int):
    sc = cfg.scenario
    state_dir = Path(sc.state_dir)
    manifest_path = state_dir / "manifest.jsonl"
    if not sc.enabled or not manifest_path.exists():
        return None, None
    records = load_manifest(manifest_path, base_dir=state_dir)
    if not records:
        return None, None
    library = ScenarioLibrary(records)
    sampler = ScenarioSampler(
        library,
        SamplerConfig(
            anchor_fraction=sc.anchor_fraction,
            loss_state_fraction=sc.loss_state_fraction,
            stratum_floor=sc.stratum_floor,
            retention_boost=sc.retention_boost,
            noop_min=sc.noop_min,
            noop_max=sc.noop_max,
            promote_winrate=getattr(sc, "promote_winrate", 0.45),
            promote_min_episodes=getattr(sc, "promote_min_episodes", 15),
            zpd_threshold=getattr(sc, "zpd_threshold", 0.20),
            zpd_late_bias=getattr(sc, "zpd_late_bias", 0.65),
        ),
        natural_counts=Counter(r.cell_id for r in records if r.split == "train"),
        seed=seed,
    )
    probe_rng = np.random.default_rng(seed + 17)

    def provider(_i: int):
        if sc.probe_fraction > 0 and probe_rng.random() < sc.probe_fraction:
            assignment = sampler.sample_probe()
            if assignment.kind == "probe":
                return assignment
        return sampler.sample()
    return sampler, provider


def run_hierarchical_training(
    cfg,
    args,
    *,
    build_workers: Callable,
    seed_everything: Callable,
    ensure_log: Callable,
    prune_checkpoints: Callable,
) -> None:
    hc = cfg.hierarchical
    if not hc.enabled:
        raise SystemExit("hierarchical.enabled=false in config; enable it to use --algo hierarchical")
    seed_everything(cfg.seed)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        raise RuntimeError("CUDA is requested but unavailable. Set device='cpu' for smoke testing.")
    device = torch.device(cfg.device)
    print(f"PyTorch: {torch.__version__}\nDevice: {device}")
    model = HierarchicalActorCritic(
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
    ).to(device)
    if getattr(args, "init_from", ""):
        payload = torch.load(args.init_from, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "tennis_rl_hierarchical_v2":
            raise ValueError("--init-from must be a tennis_rl_hierarchical_v2 checkpoint")
        if tuple(payload.get("skills", ())) != model.skills:
            raise ValueError("--init-from checkpoint task order does not match configs/scenario.json")
        model.load_state_dict(payload["model"], strict=True)
        print(f"Initialized weights only from {args.init_from}; starting a fresh PPO optimizer")
    trainer = HierarchicalPPOTrainer(model, hc, device)
    if args.resume:
        print(f"Resumed hierarchy checkpoint: {trainer.load(args.resume)}")
    workers = build_workers(cfg)
    sampler, scenario_provider = _scenario_sampler(cfg, cfg.seed)
    sc = cfg.scenario
    if sc.enabled and scenario_provider is None:
        print(
            f"WARNING: scenario.enabled=true but {Path(sc.state_dir) / 'manifest.jsonl'} was not found; "
            "running on full matches until scenarios are harvested."
        )
    task_dataset = TaskAnnotationDataset(
        cfg.ppo.demo_root, model.skills, model.durations,
        button_pulse_durations=model.button_pulse_durations,
    )
    if len(task_dataset) or task_dataset.state_annotations:
        print(
            f"Explicit annotations: task_segments={len(task_dataset)} "
            f"state_frames={len(task_dataset.state_annotations)}"
        )
    else:
        print(
            "No task/state annotation sidecars found; specialists will learn from online rewards only. "
            "No ball/contact labels are inferred from pixels."
        )
    annotation_rng = np.random.default_rng(cfg.seed + 991)
    env = ParallelTennis(
        workers,
        stack_size=cfg.environment.stack_size,
        max_steps=cfg.environment.episode_max_steps,
        point_episodes=bool(sc.enabled and sc.point_episodes),
        point_max_steps=sc.point_max_steps,
        scenario_provider=scenario_provider,
        segment_mode=bool(sc.enabled and sc.segment_mode),
        late_max_steps=sc.late_max_steps,
        mid_max_steps=sc.mid_max_steps,
        start_max_steps=sc.start_max_steps,
        serve_stall_penalty=sc.serve_stall_penalty,
        rally_progress_reward=sc.rally_progress_reward,
        late_progress_reward=sc.late_progress_reward,
    )
    if args.smoke_test:
        try:
            smoke_obs = env.reset()
            controller = HierarchicalController(model)
            for step in range(16):
                obs_tensor = torch.from_numpy(smoke_obs).to(device)
                with torch.no_grad():
                    action, pulse_frames = controller.act_with_pulse(obs_tensor)
                batch = env.step(action.cpu().numpy(), pulse_frames=pulse_frames.cpu().numpy())
                smoke_obs = batch.observations.copy()
                for i in np.flatnonzero(batch.dones):
                    smoke_obs[int(i)] = env.current_observation(int(i))
                    controller.reset(torch.tensor([j == int(i) for j in range(env.n)], device=device))
                print(f"smoke step={step + 1:02d} obs={tuple(batch.observations.shape)} actions={action.cpu().tolist()} rewards={batch.rewards.tolist()}")
        finally:
            env.close()
        return

    run_dir = Path(hc.log_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_file, csv_writer = ensure_log(run_dir, [
        "update", "env_steps", "loss", "manager_policy_loss", "expert_policy_loss",
        "manager_value_loss", "expert_value_loss", "manager_supervision_loss",
        "episodes", "point_wins", "point_losses", "service_faults", "rally_survivals"
    ])
    print(
        f"Hierarchical PPO: skills={list(model.skills)}, durations={list(model.durations)}, "
        f"workers={env.n}, rollout_steps={hc.rollout_steps}, device={device}"
    )
    observations = env.reset()
    belief_hidden = model.initial_belief(env.n, device)
    steps_done = int(trainer.env_steps)
    update = int(trainer.update_index)
    start_time = time.perf_counter()
    try:
        while steps_done < int(cfg.ppo.total_steps):
            data, observations, stats = collect_rollout(model, env, observations, cfg, device, belief_hidden)
            belief_hidden = data.pop("belief_hidden_out").to(device)
            annotation_batch = task_dataset.sample_training_batch(int(hc.batch_size), annotation_rng)
            if annotation_batch is not None:
                data.update(annotation_batch)
            metrics = trainer.update(data)
            steps_done += env.n * int(hc.rollout_steps)
            trainer.env_steps = steps_done
            update += 1
            trainer.update_index = update
            point_decisions = stats.get("point_wins", 0.0) + stats.get("point_losses", 0.0)
            point_win_rate = stats.get("point_wins", 0.0) / max(point_decisions, 1.0)
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            fps = steps_done / elapsed
            print(
                f"update={update:05d} steps={steps_done:,}/{cfg.ppo.total_steps:,} "
                f"fps={fps:.1f} loss={metrics['loss']:.4f} "
                f"manager_pol={metrics['manager_policy_loss']:.4f} "
                f"expert_pol={metrics['expert_policy_loss']:.4f} "
                f"manager_sup={metrics['manager_supervision_loss']:.4f} "
                f"points={int(point_decisions)} point_win_rate={point_win_rate:.3f} "
                f"skills={int(stats.get('manager_decisions', 0))} faults={int(stats.get('service_faults', 0))}",
                flush=True,
            )
            csv_writer.writerow([
                update, steps_done, metrics["loss"], metrics["manager_policy_loss"],
                metrics["expert_policy_loss"], metrics["manager_value_loss"],
                metrics["expert_value_loss"], metrics["manager_supervision_loss"],
                stats.get("episodes", 0), stats.get("point_wins", 0), stats.get("point_losses", 0),
                stats.get("service_faults", 0), stats.get("rally_survivals", 0),
            ])
            csv_file.flush()
            if update % max(1, int(hc.checkpoint_every_updates)) == 0:
                path = Path(hc.save_dir) / f"hierarchical_step_{steps_done}.pt"
                trainer.save(str(path), {"config": _jsonable(cfg), "point_win_rate_this_rollout": point_win_rate})
                prune_checkpoints(Path(hc.save_dir), hc.max_step_checkpoints, prefix="hierarchical_step_")
                trainer.save(str(Path(hc.save_dir) / "hierarchical_latest.pt"), {"config": _jsonable(cfg)})
            if args.max_updates is not None and update >= args.max_updates:
                print(f"Reached max_updates ({args.max_updates}), terminating.")
                trainer.save(str(Path(hc.save_dir) / "hierarchical_latest.pt"), {"config": _jsonable(cfg)})
                break
    finally:
        csv_file.close()
        env.close()


def _jsonable(obj: Any) -> Any:
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonable(v) for v in obj]
    return obj
