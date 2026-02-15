"""Training loop/orchestration for Knockoff PPO."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .knockoff_ppo_args import Args
from .knockoff_ppo_support import (
    PATCH_ID,
    build_policy_payload,
    find_latest_checkpoint_row,
    load_upstream_lock,
    next_experiment_dir,
    resolve_torch_device,
    torch_load_dict,
)
from .checkpoint_manager import CheckpointManager, CheckpointManagerConfig
from .cpp_env import CppClashEnvBatch
from .env_layout import as_card_position_masks, parse_placement_runtime
from .masked_policy_adapter import MaskedPPOAgent, MaskedPolicyConfig, PlacementConfig
from .ppo import branch_sizes_from_action_space
from .selfplay import (
    EloRatingTracker,
    EloRatingTrackerConfig,
    OpponentModelShape,
    OpponentPolicyCache,
    SelfPlayPoolConfig,
    SelfPlayPoolManager,
)


def _as_strict_vector(x: np.ndarray, expected_size: int, label: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"{label} size mismatch expected={expected_size} got={arr.size}")
    return arr


def _is_selfplay_controller(value: str) -> bool:
    normalized = str(value).strip().lower()
    return normalized in {"selfplay", "selfplay_pool"}


def _ansi(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


class _TrainingVisualizer:
    def __init__(self, fps: int):
        from training.tools.cpp_view_renderer import ArenaRenderer, kv

        self.fps = max(1, int(fps))
        self.env_index = 0
        self._kv = kv
        self.renderer = ArenaRenderer(title="knockoff_cr training live view", width=1100, height=620, headless=False)
        self.collapsed = False

    def update(self, states: list[dict[str, Any]], metrics: dict[str, Any]) -> bool:
        events = self.renderer.poll_events(esc_quit=False)
        if events["quit"]:
            return False
        if events["esc"]:
            self.collapsed = not self.collapsed
        if states:
            if events["left"]:
                self.env_index = (self.env_index - 1) % len(states)
            if events["right"]:
                self.env_index = (self.env_index + 1) % len(states)
            self.env_index = max(0, min(self.env_index, len(states) - 1))
            state = states[self.env_index]
        else:
            state = {"entities": [], "elixir": [0.0, 0.0]}

        elixir = state.get("elixir", [0.0, 0.0])
        entities = state.get("entities", [])
        if self.collapsed:
            lines = [
                "[TRAINING VIEW]",
                "Collapsed mode: training running in background",
                self._kv("Global step", str(int(metrics.get("global_step", 0)))),
                self._kv("Iteration", str(int(metrics.get("iteration", 0)))),
                self._kv("SPS", str(int(metrics.get("sps", 0)))),
                self._kv("Env", str(self.env_index)),
                "",
                "Press Esc to restore full preview",
                "Left/Right: switch env",
                "Close window to disable viewer",
            ]
            self.renderer.draw({"entities": [], "elixir": [0.0, 0.0]}, lines, fps_limit=5)
            return True

        lines = [
            "[TRAINING]",
            self._kv("Env index", str(self.env_index)),
            self._kv("Global step", str(int(metrics.get("global_step", 0)))),
            self._kv("Iteration", str(int(metrics.get("iteration", 0)))),
            self._kv("SPS", str(int(metrics.get("sps", 0)))),
            self._kv("FPS target", str(self.fps)),
            "",
            "[ENV]",
            self._kv("Sim time", f"{float(state.get('sim_time_s', 0.0)):.2f}s"),
            self._kv("Done", str(bool(state.get("done", False)))),
            self._kv("Truncated", str(bool(state.get("truncation", False)))),
            self._kv("Entities", str(len(entities))),
            self._kv("Pending", str(len(state.get("pending_spawns", [])))),
            self._kv("Elixir P0", f"{float(elixir[0]):.2f}"),
            self._kv("Elixir P1", f"{float(elixir[1]):.2f}"),
            "",
            "Esc: Collapse preview",
            "Left/Right: Switch env",
            "Close window: disable viewer",
        ]
        stats = self.renderer.draw(state, lines, fps_limit=self.fps)
        _ = stats
        return True

    def close(self) -> None:
        self.renderer.close()

def run_training(args: Args) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be > 0")

    upstream = load_upstream_lock()

    run_root = Path(args.run_dir)
    resume_row: dict[str, Any] | None = None
    if args.resume_latest:
        experiment_dir, resume_row = find_latest_checkpoint_row(run_root)
    else:
        experiment_dir = next_experiment_dir(run_root)
        experiment_dir.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    derived_physics_tps = max(1, int(args.cpp_tick_hz))
    # C++ backend keeps 1 PPO step == 1 simulated second by default.
    ticks_per_step = max(1, int(args.cpp_tick_hz))
    sim_seconds_per_step = float(ticks_per_step) / float(derived_physics_tps)

    tb_dir = Path(args.tb_dir) if args.tb_dir else experiment_dir / "tensorboard"
    writer = SummaryWriter(str(tb_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = resolve_torch_device(use_cuda=args.cuda, use_mps=args.mps)
    ckpt = CheckpointManager(CheckpointManagerConfig(run_dir=str(experiment_dir)))

    run_wall_start = time.perf_counter()
    startup_spawn_seconds = 0.0
    startup_reset_seconds = 0.0
    training_wall_seconds = 0.0
    rollout_wall_seconds_total = 0.0
    update_wall_seconds_total = 0.0
    iteration_wall_seconds_total = 0.0
    completed_iterations = 0

    batch_env: CppClashEnvBatch | None = None
    live_view: _TrainingVisualizer | None = None
    live_view_failed = False
    try:
        num_envs = int(args.num_envs)
        startup_spawn_start = time.perf_counter()
        batch_env = CppClashEnvBatch(
            num_envs=num_envs,
            tick_hz=int(args.cpp_tick_hz),
            max_sim_seconds=float(args.cpp_max_sim_seconds),
            seed=int(args.seed),
            num_threads=int(args.cpp_num_threads),
        )
        startup_spawn_seconds = time.perf_counter() - startup_spawn_start
        if bool(args.visualize):
            try:
                live_view = _TrainingVisualizer(fps=int(args.visualize_fps))
                print(f"[PPO] live viewer enabled env=0 fps={int(args.visualize_fps)} (Left/Right to switch env)")
            except Exception as exc:
                live_view_failed = True
                print(f"[PPO] live viewer disabled: {exc}")

        action_order = batch_env.action_order
        agent_keys = batch_env.agent_keys
        stream_team_indices: list[int] = []
        action_index = {name: idx for idx, name in enumerate(action_order)}
        wait_action_template = np.zeros((len(action_order),), dtype=np.int32)
        if "wait" in action_index:
            wait_action_template[action_index["wait"]] = 1
        per_agent_action_space = batch_env.action_space[agent_keys[0]]
        branch_sizes = branch_sizes_from_action_space(per_agent_action_space.spaces, action_order)
        num_branches = len(branch_sizes)

        supported_controllers = {"external", "human", "selfplay", "selfplay_pool"}
        team_controllers = [str(args.team0_controller).lower(), str(args.team1_controller).lower()]
        for team_idx, controller in enumerate(team_controllers):
            if controller not in supported_controllers:
                raise ValueError(
                    f"Unsupported controller '{controller}' for team {team_idx}. "
                    f"Supported={sorted(supported_controllers)}"
                )

        selfplay_teams = [idx for idx, ctrl in enumerate(team_controllers) if _is_selfplay_controller(ctrl)]
        selfplay_enabled = bool(args.selfplay_enabled) or bool(selfplay_teams)
        if bool(args.selfplay_enabled) and not selfplay_teams:
            raise ValueError("selfplay_enabled=true requires at least one team controller set to selfplay_pool")
        if len(selfplay_teams) > 1:
            raise ValueError("Only one selfplay_pool team is supported")
        selfplay_team = selfplay_teams[0] if selfplay_teams else None

        selfplay_pool: SelfPlayPoolManager | None = None
        elo_tracker: EloRatingTracker | None = None
        elo_trained_team: int | None = None
        env_selfplay_selection: list[Any] = [None for _ in range(num_envs)]
        selfplay_sample_counts: dict[str, int] = {"latest": 0, "recent": 0, "anchor": 0, "none": 0}
        if selfplay_enabled:
            selfplay_pool = SelfPlayPoolManager(
                SelfPlayPoolConfig(
                    root_dir=str(experiment_dir / "selfplay_pool"),
                    recent_capacity=int(args.selfplay_recent_capacity),
                    anchor_capacity=int(args.selfplay_anchor_capacity),
                    anchor_every=int(args.selfplay_anchor_every),
                    latest_prob=float(args.selfplay_latest_prob),
                    recent_prob=float(args.selfplay_recent_prob),
                    anchor_prob=float(args.selfplay_anchor_prob),
                )
            )
            print(
                f"[PPO] selfplay enabled team={selfplay_team} recent={args.selfplay_recent_capacity} "
                f"anchors={args.selfplay_anchor_capacity} bootstrap=mirror_current"
            )
            writer.add_text(
                "selfplay/config",
                f"team={selfplay_team} recent={args.selfplay_recent_capacity} "
                f"anchors={args.selfplay_anchor_capacity} anchor_every={args.selfplay_anchor_every} "
                "bootstrap=mirror_current",
            )
            if bool(args.selfplay_elo_enabled):
                elo_tracker = EloRatingTracker(
                    EloRatingTrackerConfig(
                        root_dir=str(experiment_dir / "selfplay_pool"),
                        initial_rating=float(args.selfplay_elo_initial),
                        k_factor=float(args.selfplay_elo_k),
                    )
                )
                if selfplay_team is not None:
                    elo_trained_team = 1 - int(selfplay_team)
                print(
                    f"[PPO] selfplay elo enabled initial={args.selfplay_elo_initial} "
                    f"k={args.selfplay_elo_k} trained_team={elo_trained_team}"
                )
                writer.add_text(
                    "selfplay/elo_config",
                    f"enabled=true initial={args.selfplay_elo_initial} k={args.selfplay_elo_k} "
                    f"trained_team={elo_trained_team}",
                )

        def _refresh_selfplay_selection(env_idx: int) -> None:
            if not selfplay_enabled or selfplay_team is None or selfplay_pool is None:
                return
            selection = selfplay_pool.sample()
            env_selfplay_selection[env_idx] = selection
            selfplay_sample_counts[selection.category] = selfplay_sample_counts.get(selection.category, 0) + 1

        def _controllers_for_env(env_idx: int) -> list[str]:
            controllers = team_controllers.copy()
            if selfplay_enabled and selfplay_team is not None and _is_selfplay_controller(controllers[selfplay_team]):
                controllers[selfplay_team] = "external"
            return controllers

        stream_env: list[int] = []
        stream_agent: list[str] = []
        for env_i in range(num_envs):
            for agent_key in agent_keys:
                team_idx = int(agent_key.split("_")[1])
                controller = team_controllers[team_idx]
                if controller == "external" or _is_selfplay_controller(controller):
                    stream_env.append(env_i)
                    stream_agent.append(agent_key)
                    stream_team_indices.append(team_idx)
        if not stream_env:
            raise ValueError("No trainable streams; set at least one team controller to external")

        batch_size = int(len(stream_env) * args.num_steps)
        if batch_size <= 0:
            raise ValueError("Computed batch_size must be > 0")
        if (
            device.type == "mps"
            and bool(args.prefer_cpu_for_small_batches)
            and batch_size <= int(args.small_batch_threshold)
        ):
            prev_device = str(device)
            device = torch.device("cpu")
            print(
                f"[PPO] device fallback {prev_device}->cpu for small batch_size={batch_size} "
                f"(threshold={int(args.small_batch_threshold)})"
            )
        minibatch_size = max(1, int(batch_size // max(1, int(args.num_minibatches))))
        num_iterations = max(1, int(math.ceil(float(args.total_timesteps) / float(batch_size))))

        startup_reset_start = time.perf_counter()
        for env_idx in range(num_envs):
            _refresh_selfplay_selection(env_idx)
        startup_options = [
            {
                "team_controllers": _controllers_for_env(env_idx),
                "training_mode": bool(args.training_mode),
                "ticks_per_step": int(ticks_per_step),
            }
            for env_idx in range(num_envs)
        ]
        startup_seeds = [int(args.seed) + env_idx for env_idx in range(num_envs)]
        reset_obs, _ = batch_env.reset_many(
            seeds=startup_seeds,
            options_per_env=startup_options,
        )
        for env_idx in range(num_envs):
            obs_entry = reset_obs[env_idx]
            if obs_entry is None:
                raise RuntimeError(f"startup reset did not return observation for env {env_idx}")
        startup_reset_seconds = time.perf_counter() - startup_reset_start
        startup_total_seconds = startup_spawn_seconds + startup_reset_seconds
        print(
            f"[PPO] startup num_envs={num_envs} stream_count={len(stream_env)} "
            f"spawn_s={startup_spawn_seconds:.3f} reset_s={startup_reset_seconds:.3f} total_s={startup_total_seconds:.3f}"
        )
        writer.add_scalar("timing/startup_spawn_seconds", startup_spawn_seconds, 0)
        writer.add_scalar("timing/startup_reset_seconds", startup_reset_seconds, 0)
        writer.add_scalar("timing/startup_total_seconds", startup_total_seconds, 0)

        sample_env = int(stream_env[0])
        sample_team = int(stream_team_indices[0])
        sample_key = f"agent_{sample_team}"
        sample_obs = reset_obs[sample_env]
        if sample_obs is None:
            raise RuntimeError(f"startup reset did not return sample observation for env {sample_env}")
        obs_dim = int(len(sample_obs[sample_key]["vector"]))
        mask_dim = int(len(sample_obs[sample_key]["action_mask"]))
        placement_runtime = parse_placement_runtime(batch_env.spec_data)
        card_mask_shape = as_card_position_masks(
            sample_obs[sample_key],
            expected_position_count=placement_runtime.position_count,
        ).shape
        if len(card_mask_shape) != 2:
            raise ValueError(f"Expected position_masks_for_all_cards 2D shape, got {card_mask_shape}")
        cards_count, position_count = int(card_mask_shape[0]), int(card_mask_shape[1])
        if placement_runtime.position_count != position_count:
            raise ValueError(
                f"placement_schema position_count mismatch expected={position_count} got={placement_runtime.position_count}"
            )
        hidden_sizes = tuple(int(x.strip()) for x in str(args.hidden_sizes).split(",") if x.strip())
        if not hidden_sizes:
            raise ValueError("--hidden-sizes produced an empty layer list")
        selfplay_agent_key = f"agent_{selfplay_team}" if selfplay_team is not None else None
        selfplay_policy_cache: OpponentPolicyCache | None = None
        if selfplay_enabled and selfplay_team is not None:
            selfplay_policy_cache = OpponentPolicyCache(
                shape=OpponentModelShape(
                    obs_dim=obs_dim,
                    action_branch_sizes=tuple(int(x) for x in branch_sizes),
                    action_order=tuple(action_order),
                    placement=PlacementConfig(
                        grid_width=int(placement_runtime.config.grid_width),
                        grid_height=int(placement_runtime.config.grid_height),
                        region_cols=int(placement_runtime.config.region_cols),
                        region_rows=int(placement_runtime.config.region_rows),
                        region_cell_count=int(placement_runtime.config.region_cell_count),
                    ),
                ),
                device=device,
                deterministic=bool(args.selfplay_deterministic),
                max_cached=int(args.selfplay_max_cached_policies),
            )

        stream_env_np = np.asarray(stream_env, dtype=np.int64)
        stream_team_np = np.asarray(stream_team_indices, dtype=np.int64)
        current_obs_vectors = np.zeros((num_envs, len(agent_keys), obs_dim), dtype=np.float32)
        current_action_masks = np.zeros((num_envs, len(agent_keys), mask_dim), dtype=np.float32)
        current_card_masks = np.zeros((num_envs, len(agent_keys), cards_count, position_count), dtype=np.float32)

        def _write_reset_obs_for_env(env_idx: int, obs_entry: dict[str, Any]) -> None:
            for team_idx in range(len(agent_keys)):
                agent_key = f"agent_{team_idx}"
                current_obs_vectors[env_idx, team_idx, :] = _as_strict_vector(
                    obs_entry[agent_key]["vector"], obs_dim, f"obs[{env_idx}][{agent_key}]"
                )
                current_action_masks[env_idx, team_idx, :] = _as_strict_vector(
                    obs_entry[agent_key]["action_mask"], mask_dim, f"mask[{env_idx}][{agent_key}]"
                )
                current_card_masks[env_idx, team_idx, :, :] = as_card_position_masks(
                    obs_entry[agent_key],
                    expected_position_count=placement_runtime.position_count,
                )

        for env_idx in range(num_envs):
            obs_entry = reset_obs[env_idx]
            if obs_entry is None:
                raise RuntimeError(f"startup reset did not return observation for env {env_idx}")
            _write_reset_obs_for_env(env_idx, obs_entry)

        def _act_with_current_policy_arrays(env_idx: int, team_idx: int) -> dict[str, int]:
            vec = torch.from_numpy(current_obs_vectors[env_idx, team_idx]).unsqueeze(0).to(device)
            mask = torch.from_numpy(current_action_masks[env_idx, team_idx]).unsqueeze(0).to(device)
            card_pos_masks = torch.from_numpy(current_card_masks[env_idx, team_idx]).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mirror, _, _, _, _, _ = agent.get_action_and_value(
                    vec,
                    mask,
                    card_pos_masks,
                    deterministic=bool(args.selfplay_deterministic),
                )
            branches = action_mirror[0].detach().cpu().tolist()
            return {key: int(branches[i]) for i, key in enumerate(action_order)}

        def _refresh_stream_tensors_from_packed() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            obs_np = current_obs_vectors[stream_env_np, stream_team_np, :]
            mask_np = current_action_masks[stream_env_np, stream_team_np, :]
            card_np = current_card_masks[stream_env_np, stream_team_np, :, :]
            return (
                torch.as_tensor(obs_np, dtype=torch.float32, device=device),
                torch.as_tensor(mask_np, dtype=torch.float32, device=device),
                torch.as_tensor(card_np, dtype=torch.float32, device=device),
            )

        agent = MaskedPPOAgent(
            MaskedPolicyConfig(
                obs_dim=obs_dim,
                action_branch_sizes=branch_sizes,
                action_order=tuple(action_order),
                hidden_sizes=hidden_sizes,
                placement=placement_runtime.config,
            )
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        global_step = 0
        start_iteration = 1
        log_every_steps = int(args.log_every)
        flush_every_logs = max(1, int(args.flush_every_logs))
        next_log_step = global_step + max(1, log_every_steps) if log_every_steps > 0 else global_step
        next_ckpt_step = int(args.ckpt_every) if args.ckpt_every > 0 else None
        last_ckpt_saved_step: int | None = None
        pending_logs = 0
        start_time = time.time()

        next_obs, next_mask, next_card_masks = _refresh_stream_tensors_from_packed()
        next_done = torch.zeros(len(stream_env), dtype=torch.float32, device=device)

        if args.resume_latest and resume_row is not None:
            train_state_path = Path(resume_row.get("train_state", ""))
            state = torch_load_dict(train_state_path, label="train_state", map_location=device)
            agent.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            # Honor current CLI learning-rate when resuming, regardless of stored optimizer LR.
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(args.learning_rate)
            global_step = int(state.get("global_step", 0))
            start_iteration = int(state.get("iteration", 0)) + 1
            if log_every_steps > 0:
                next_log_step = ((global_step // log_every_steps) + 1) * log_every_steps
            if next_ckpt_step is not None:
                next_ckpt_step = ((global_step // int(args.ckpt_every)) + 1) * int(args.ckpt_every)
            print(f"[PPO] resumed iteration={start_iteration} global_step={global_step}")

        if args.resume_latest:
            remaining_steps = max(0, int(args.total_timesteps) - int(global_step))
            num_iterations = int(math.ceil(float(remaining_steps) / float(batch_size))) if remaining_steps > 0 else 0

        obs = torch.zeros((args.num_steps, len(stream_env), obs_dim), dtype=torch.float32, device=device)
        masks = torch.zeros((args.num_steps, len(stream_env), mask_dim), dtype=torch.float32, device=device)
        card_masks = torch.zeros(
            (args.num_steps, len(stream_env), cards_count, position_count), dtype=torch.float32, device=device
        )
        region_masks_used = torch.zeros(
            (args.num_steps, len(stream_env), branch_sizes[action_order.index("position_region")]),
            dtype=torch.float32,
            device=device,
        )
        cell_masks_used = torch.zeros(
            (args.num_steps, len(stream_env), branch_sizes[action_order.index("position_cell")]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.zeros((args.num_steps, len(stream_env), num_branches), dtype=torch.long, device=device)
        logprobs = torch.zeros((args.num_steps, len(stream_env)), dtype=torch.float32, device=device)
        rewards = torch.zeros((args.num_steps, len(stream_env)), dtype=torch.float32, device=device)
        dones = torch.zeros((args.num_steps, len(stream_env)), dtype=torch.float32, device=device)
        values = torch.zeros((args.num_steps, len(stream_env)), dtype=torch.float32, device=device)
        train_mask = torch.ones((args.num_steps, len(stream_env)), dtype=torch.float32, device=device)

        env_to_stream_indices: dict[int, list[int]] = {env_i: [] for env_i in range(num_envs)}
        for s, env_i in enumerate(stream_env):
            env_to_stream_indices[env_i].append(s)
        external_teams = sorted({int(agent_key.split("_")[1]) for agent_key in stream_agent})

        episode_returns = np.zeros((num_envs, len(agent_keys)), dtype=np.float64)
        episode_lengths = np.zeros((num_envs,), dtype=np.int64)
        episodes_seen = np.zeros((len(agent_keys),), dtype=np.int64)
        episodes_won = np.zeros((len(agent_keys),), dtype=np.int64)

        def _save_checkpoint(step: int, iteration: int, note: str) -> None:
            nonlocal last_ckpt_saved_step
            step_dir = ckpt.save(
                step=step,
                train_state={
                    "global_step": int(step),
                    "iteration": int(iteration),
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "upstream_repo": upstream["repo"],
                    "upstream_path": upstream["path"],
                    "upstream_commit": upstream["commit"],
                    "upstream_sha256": upstream["sha256"],
                    "patch_id": PATCH_ID,
                },
                policy_state=build_policy_payload(agent.model, upstream=upstream),
                metadata={"note": note, "iteration": int(iteration)},
            )
            if selfplay_pool is not None:
                registered = selfplay_pool.register_checkpoint(
                    step=int(step),
                    policy_source_path=step_dir / "policy_only.pt",
                )
                if elo_tracker is not None:
                    elo_tracker.ensure_checkpoint(int(registered.step))
                    elo_tracker.flush()
            last_ckpt_saved_step = int(step)

        training_wall_start = time.perf_counter()
        if num_iterations == 0:
            print(
                f"[PPO] target total_timesteps={args.total_timesteps} already reached at global_step={global_step}; "
                "skipping training loop"
            )
        for iteration in range(start_iteration, start_iteration + num_iterations):
                run_iteration = iteration - start_iteration + 1
                iteration_wall_start = time.perf_counter()
                if args.anneal_lr:
                    frac = 1.0 - (run_iteration - 1.0) / max(1, num_iterations)
                    optimizer.param_groups[0]["lr"] = frac * args.learning_rate

                rollout_wall_start = time.perf_counter()
                for step in range(args.num_steps):
                    global_step += len(stream_env)
                    obs[step] = next_obs
                    masks[step] = next_mask
                    card_masks[step] = next_card_masks
                    dones[step] = next_done

                    with torch.no_grad():
                        action, logprob, _, value, used_region_mask, used_cell_mask = agent.get_action_and_value(
                            next_obs, next_mask, next_card_masks
                        )
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob
                    region_masks_used[step] = used_region_mask
                    cell_masks_used[step] = used_cell_mask

                    action_np = action.detach().cpu().numpy().astype(np.int32, copy=False)
                    actions_per_env = np.empty((num_envs, len(agent_keys), num_branches), dtype=np.int32)
                    actions_per_env[:] = wait_action_template
                    for s in range(len(stream_env)):
                        env_i = stream_env[s]
                        team_idx = stream_team_indices[s]
                        actions_per_env[env_i, team_idx, :] = action_np[s, :]
                    if selfplay_enabled and selfplay_agent_key is not None and selfplay_policy_cache is not None:
                        selfplay_team_idx = int(selfplay_agent_key.split("_")[1])
                        for env_i in range(num_envs):
                            selection = env_selfplay_selection[env_i]
                            opponent_action: dict[str, int]
                            if selection is not None and selection.checkpoint is not None:
                                try:
                                    opponent = selfplay_policy_cache.get(selection.checkpoint.policy_path)
                                    opponent_action = opponent.act_arrays(
                                        vec=current_obs_vectors[env_i, selfplay_team_idx],
                                        mask=current_action_masks[env_i, selfplay_team_idx],
                                        card_masks=current_card_masks[env_i, selfplay_team_idx],
                                    )
                                except Exception as exc:
                                    print(
                                        f"[PPO] selfplay opponent action failed env={env_i} "
                                        f"path={selection.checkpoint.policy_path}: {exc}"
                                    )
                                    opponent_action = _act_with_current_policy_arrays(env_i, selfplay_team_idx)
                            else:
                                opponent_action = _act_with_current_policy_arrays(env_i, selfplay_team_idx)
                            for key, value in opponent_action.items():
                                b_idx = action_index.get(key)
                                if b_idx is not None:
                                    actions_per_env[env_i, selfplay_team_idx, b_idx] = int(value)

                    packed_obs, packed_mask, packed_cards, packed_reward, packed_done, packed_trunc, packed_winner, packed_rt = (
                        batch_env.step_many_packed(actions_per_env)
                    )
                    current_obs_vectors = packed_obs
                    current_action_masks = packed_mask
                    current_card_masks = packed_cards

                    stream_reward = torch.zeros((len(stream_env),), dtype=torch.float32, device=device)
                    stream_done = torch.zeros((len(stream_env),), dtype=torch.float32, device=device)
                    step_roundtrip_s = float(packed_rt) / float(max(1, num_envs))
                    envs_to_reset: list[int] = []
                    for env_idx in range(num_envs):
                        reward_np = np.asarray(packed_reward[env_idx], dtype=np.float64)
                        if reward_np.shape[0] == len(agent_keys):
                            episode_returns[env_idx] += reward_np
                        episode_lengths[env_idx] += 1
                        done_flag = bool(packed_done[env_idx] or packed_trunc[env_idx])
                        if done_flag:
                            team0_return = float(episode_returns[env_idx, 0])
                            team1_return = float(episode_returns[env_idx, 1])
                            ep_len = int(episode_lengths[env_idx])
                            writer.add_scalar("charts/episode_return_team0", team0_return, global_step)
                            writer.add_scalar("charts/episode_return_team1", team1_return, global_step)
                            writer.add_scalar("charts/episode_length", ep_len, global_step)

                            external_returns = [float(episode_returns[env_idx, t]) for t in external_teams]
                            if external_returns:
                                writer.add_scalar(
                                    "charts/episode_return_external_mean",
                                    float(np.mean(external_returns)),
                                    global_step,
                                )

                            winner_raw = int(packed_winner[env_idx])
                            winner = winner_raw if winner_raw >= 0 else None
                            if winner is not None:
                                for team_i in range(len(agent_keys)):
                                    episodes_seen[team_i] += 1
                                episodes_won[int(winner)] += 1
                                if episodes_seen[0] > 0:
                                    writer.add_scalar(
                                        "charts/win_rate_team0",
                                        float(episodes_won[0]) / float(episodes_seen[0]),
                                        global_step,
                                    )
                                if episodes_seen[1] > 0:
                                    writer.add_scalar(
                                        "charts/win_rate_team1",
                                        float(episodes_won[1]) / float(episodes_seen[1]),
                                        global_step,
                                    )
                                if len(external_teams) == 1:
                                    t = external_teams[0]
                                    if episodes_seen[t] > 0:
                                        writer.add_scalar(
                                            "charts/win_rate_external",
                                            float(episodes_won[t]) / float(episodes_seen[t]),
                                            global_step,
                                        )

                            selection_for_episode = env_selfplay_selection[env_idx]
                            if (
                                elo_tracker is not None
                                and elo_trained_team is not None
                                and selection_for_episode is not None
                                and selection_for_episode.checkpoint is not None
                            ):
                                if winner is None:
                                    score_current = 0.5
                                elif int(winner) == int(elo_trained_team):
                                    score_current = 1.0
                                else:
                                    score_current = 0.0
                                current_elo, opp_elo = elo_tracker.record_match_vs_checkpoint(
                                    checkpoint_step=int(selection_for_episode.checkpoint.step),
                                    score_current=float(score_current),
                                )
                                writer.add_scalar("selfplay/elo_current", float(current_elo), global_step)
                                writer.add_scalar("selfplay/elo_opponent", float(opp_elo), global_step)
                                writer.add_scalar("selfplay/elo_games", float(elo_tracker.total_games), global_step)

                            _refresh_selfplay_selection(env_idx)
                            envs_to_reset.append(env_idx)
                            episode_returns[env_idx, :] = 0.0
                            episode_lengths[env_idx] = 0
                        for s in env_to_stream_indices[env_idx]:
                            agent_key = stream_agent[s]
                            team_idx = int(agent_key.split("_")[1])
                            stream_reward[s] = float(packed_reward[env_idx, team_idx])
                            stream_done[s] = 1.0 if done_flag else 0.0
                            is_trainable_sample = True
                            if selfplay_enabled and selfplay_team is not None and team_idx == int(selfplay_team):
                                selection = env_selfplay_selection[env_idx]
                                is_trainable_sample = bool(selection is None or selection.checkpoint is None)
                            train_mask[step, s] = 1.0 if is_trainable_sample else 0.0
                    if envs_to_reset:
                        reset_options: list[dict[str, Any] | None] = [None for _ in range(num_envs)]
                        reset_seeds: list[int | None] = [None for _ in range(num_envs)]
                        for env_idx in envs_to_reset:
                            reset_options[env_idx] = {
                                "team_controllers": _controllers_for_env(env_idx),
                                "training_mode": bool(args.training_mode),
                                "ticks_per_step": int(ticks_per_step),
                            }
                        reset_obs, _ = batch_env.reset_many(
                            seeds=reset_seeds,
                            options_per_env=reset_options,
                        )
                        for env_idx in envs_to_reset:
                            obs_entry = reset_obs[env_idx]
                            if obs_entry is None:
                                raise RuntimeError(f"reset_many did not return observation for env {env_idx}")
                            _write_reset_obs_for_env(env_idx, obs_entry)
                    if live_view is not None and not live_view_failed:
                        try:
                            states = batch_env.debug_state_many()
                            keep_open = live_view.update(
                                states,
                                {
                                    "global_step": global_step,
                                    "iteration": run_iteration,
                                    "sps": int(global_step / max(1e-9, time.time() - start_time)),
                                },
                            )
                            if not keep_open:
                                live_view.close()
                                live_view = None
                        except Exception as exc:
                            live_view_failed = True
                            if live_view is not None:
                                live_view.close()
                                live_view = None
                            print(f"[PPO] live viewer disabled after runtime error: {exc}")

                    rewards[step] = stream_reward
                    should_log_step_debug = (log_every_steps <= 0) or (global_step >= next_log_step)
                    if should_log_step_debug:
                        rt_ms = step_roundtrip_s * 1000.0
                        writer.add_scalar("timing/env_step_roundtrip_ms_p50", rt_ms, global_step)
                        writer.add_scalar("timing/env_step_roundtrip_ms_p95", rt_ms, global_step)
                        writer.add_scalar("timing/env_step_roundtrip_ms_max", rt_ms, global_step)
                    next_done = stream_done
                    next_obs, next_mask, next_card_masks = _refresh_stream_tensors_from_packed()
                rollout_wall_seconds = time.perf_counter() - rollout_wall_start

                update_wall_start = time.perf_counter()
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0.0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = (
                            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        )
                    returns = advantages + values

                b_obs = obs.reshape((-1, obs_dim))
                b_masks = masks.reshape((-1, mask_dim))
                b_card_masks = card_masks.reshape((-1, cards_count, position_count))
                b_region_masks = region_masks_used.reshape((-1, region_masks_used.shape[-1]))
                b_cell_masks = cell_masks_used.reshape((-1, cell_masks_used.shape[-1]))
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1, num_branches))
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)
                b_train_mask = train_mask.reshape(-1)
                active_inds_t = torch.nonzero(b_train_mask > 0.5, as_tuple=False).reshape(-1)
                if active_inds_t.numel() == 0:
                    continue

                b_inds = active_inds_t.detach().cpu().numpy()
                active_batch_size = int(b_inds.shape[0])
                local_minibatch_size = max(1, int(active_batch_size // max(1, int(args.num_minibatches))))
                clipfracs: list[float] = []
                approx_kl = torch.tensor(0.0, device=device)
                old_approx_kl = torch.tensor(0.0, device=device)
                for _epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, active_batch_size, local_minibatch_size):
                        end = start + local_minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue, _rm, _cm = agent.get_action_and_value(
                            b_obs[mb_inds],
                            b_masks[mb_inds],
                            b_card_masks[mb_inds],
                            b_actions.long()[mb_inds],
                            b_region_masks[mb_inds],
                            b_cell_masks[mb_inds],
                        )
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            adv_mean = mb_advantages.mean()
                            # Use population std so single-sample minibatches normalize to zero
                            # instead of producing NaN from the unbiased estimator.
                            adv_std = mb_advantages.std(unbiased=False)
                            if torch.isfinite(adv_mean) and torch.isfinite(adv_std):
                                mb_advantages = (mb_advantages - adv_mean) / adv_std.clamp_min(1e-8)
                            else:
                                raise RuntimeError(
                                    f"Non-finite advantage stats detected: mean={adv_mean.item()} std={adv_std.item()}"
                                )

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        newvalue = newvalue.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                            )
                            v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break
                update_wall_seconds = time.perf_counter() - update_wall_start
                iteration_wall_seconds = time.perf_counter() - iteration_wall_start
                rollout_wall_seconds_total += rollout_wall_seconds
                update_wall_seconds_total += update_wall_seconds
                iteration_wall_seconds_total += iteration_wall_seconds
                completed_iterations += 1

                y_pred = b_values[active_inds_t].detach().cpu().numpy()
                y_true = b_returns[active_inds_t].detach().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                should_log = (log_every_steps <= 0) or (global_step >= next_log_step)
                if should_log:
                    elapsed = max(1e-9, time.time() - start_time)
                    sps = int(global_step / elapsed)
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("charts/sps", sps, global_step)
                    writer.add_scalar("charts/sim_seconds_per_wall_second", sps * sim_seconds_per_step, global_step)
                    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                    writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs) if clipfracs else 0.0), global_step)
                    writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
                    writer.add_scalar("timing/iteration_seconds", iteration_wall_seconds, global_step)
                    writer.add_scalar("timing/rollout_seconds", rollout_wall_seconds, global_step)
                    writer.add_scalar("timing/update_seconds", update_wall_seconds, global_step)
                    rollout_stream_sps = (len(stream_env) * args.num_steps) / max(1e-9, rollout_wall_seconds)
                    writer.add_scalar("timing/rollout_stream_steps_per_second", rollout_stream_sps, global_step)

                    if selfplay_enabled:
                        total_selfplay_samples = max(1, int(sum(selfplay_sample_counts.values())))
                        for category in ("latest", "recent", "anchor", "none"):
                            writer.add_scalar(
                                f"selfplay/sample_rate_{category}",
                                float(selfplay_sample_counts.get(category, 0)) / float(total_selfplay_samples),
                                global_step,
                            )
                        if selfplay_pool is not None:
                            pool_stats = selfplay_pool.summary()
                            active_train_samples = int(active_inds_t.numel()) if "active_inds_t" in locals() else 0
                            color_by_category = {
                                "latest": "36",  # cyan
                                "recent": "33",  # yellow
                                "anchor": "35",  # magenta
                                "none": "90",    # gray
                            }
                            category_counts: dict[str, int] = {
                                "mirror": 0,
                                "latest": 0,
                                "recent": 0,
                                "anchor": 0,
                            }
                            checkpoint_counts: dict[int, int] = {}
                            env_tags: list[str] = []
                            for env_idx, selection in enumerate(env_selfplay_selection):
                                if selection is None or selection.checkpoint is None:
                                    category_counts["mirror"] += 1
                                    tag = _ansi(f"E{env_idx}:mirror", "32")
                                else:
                                    cat = str(selection.category)
                                    step = int(selection.checkpoint.step)
                                    if cat in category_counts:
                                        category_counts[cat] += 1
                                    checkpoint_counts[step] = checkpoint_counts.get(step, 0) + 1
                                    cat_color = color_by_category.get(cat, "37")
                                    tag = _ansi(f"E{env_idx}:{cat}@{step}", cat_color)
                                env_tags.append(tag)
                            elo_text = ""
                            if elo_tracker is not None:
                                elo_text = f" elo={float(elo_tracker.current_rating):.1f}"
                            top_checkpoints = sorted(
                                checkpoint_counts.items(),
                                key=lambda kv: (-kv[1], -kv[0]),
                            )[:3]
                            top_cp_text = (
                                ",".join([f"{step}x{count}" for step, count in top_checkpoints])
                                if top_checkpoints
                                else "-"
                            )
                            print(
                                _ansi("[SelfPlay]", "1;37")
                                + " "
                                + _ansi(
                                    f"pool recent={pool_stats['recent_count']} anchors={pool_stats['anchor_count']} "
                                    f"registered={pool_stats['registered_count']} latest={pool_stats['latest_step']} "
                                    f"train_samples={active_train_samples}/{batch_size} "
                                    f"env_mix m={category_counts['mirror']} l={category_counts['latest']} "
                                    f"r={category_counts['recent']} a={category_counts['anchor']} "
                                    f"top_cp={top_cp_text}{elo_text}",
                                    "34",
                                )
                            )
                            preview_limit = 12
                            preview_tags = env_tags[:preview_limit]
                            hidden = len(env_tags) - len(preview_tags)
                            preview_text = " | ".join(preview_tags)
                            if hidden > 0:
                                preview_text += " | " + _ansi(f"... (+{hidden} envs)", "90")
                            print("  " + preview_text)
                    if elo_tracker is not None:
                        games = max(1, int(elo_tracker.total_games))
                        writer.add_scalar("selfplay/elo_current", float(elo_tracker.current_rating), global_step)
                        writer.add_scalar("selfplay/elo_games", float(elo_tracker.total_games), global_step)
                        writer.add_scalar(
                            "selfplay/elo_win_rate",
                            float(elo_tracker.total_wins) / float(games),
                            global_step,
                        )
                        writer.add_scalar(
                            "selfplay/elo_draw_rate",
                            float(elo_tracker.total_draws) / float(games),
                            global_step,
                        )
                        elo_tracker.flush()

                    log_suffix = ""
                    if selfplay_enabled:
                        log_suffix = f"{log_suffix} selfplay_pool=on"

                    print(
                        f"[PPO] step={global_step} iter={run_iteration}/{num_iterations} sps={sps} "
                        f"ev={explained_var:.4f} iter_s={iteration_wall_seconds:.3f} "
                        f"rollout_s={rollout_wall_seconds:.3f} update_s={update_wall_seconds:.3f}{log_suffix}"
                    )
                    pending_logs += 1
                    if pending_logs % flush_every_logs == 0:
                        writer.flush()
                    if log_every_steps > 0:
                        while next_log_step <= global_step:
                            next_log_step += log_every_steps

                if next_ckpt_step is not None and global_step >= next_ckpt_step:
                    _save_checkpoint(global_step, iteration, "cleanrl-style ppo")
                    next_ckpt_step += int(args.ckpt_every)
        training_wall_seconds = time.perf_counter() - training_wall_start

        if global_step > 0 and last_ckpt_saved_step != int(global_step):
            _save_checkpoint(global_step, completed_iterations, "final")

        total_wall_seconds = time.perf_counter() - run_wall_start
        overall_sps = global_step / max(1e-9, training_wall_seconds)
        overall_sim_speed = overall_sps * sim_seconds_per_step
        avg_iteration_seconds = iteration_wall_seconds_total / max(1, completed_iterations)
        avg_rollout_seconds = rollout_wall_seconds_total / max(1, completed_iterations)
        avg_update_seconds = update_wall_seconds_total / max(1, completed_iterations)
        timing_summary = {
            "num_envs": int(args.num_envs),
            "stream_count": int(len(stream_env)),
            "global_step": int(global_step),
            "iterations_completed": int(completed_iterations),
            "startup_spawn_seconds": float(startup_spawn_seconds),
            "startup_reset_seconds": float(startup_reset_seconds),
            "startup_total_seconds": float(startup_spawn_seconds + startup_reset_seconds),
            "training_wall_seconds": float(training_wall_seconds),
            "total_wall_seconds": float(total_wall_seconds),
            "overall_sps": float(overall_sps),
            "overall_sim_seconds_per_wall_second": float(overall_sim_speed),
            "rollout_wall_seconds_total": float(rollout_wall_seconds_total),
            "update_wall_seconds_total": float(update_wall_seconds_total),
            "iteration_wall_seconds_total": float(iteration_wall_seconds_total),
            "avg_iteration_seconds": float(avg_iteration_seconds),
            "avg_rollout_seconds": float(avg_rollout_seconds),
            "avg_update_seconds": float(avg_update_seconds),
        }
        (experiment_dir / "timing_summary.json").write_text(
            json.dumps(timing_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        writer.add_scalar("timing/training_wall_seconds", training_wall_seconds, global_step)
        writer.add_scalar("timing/total_wall_seconds", total_wall_seconds, global_step)
        writer.add_scalar("timing/overall_sps", overall_sps, global_step)
        writer.add_scalar("timing/overall_sim_seconds_per_wall_second", overall_sim_speed, global_step)
        print(
            f"[PPO] timing_summary num_envs={args.num_envs} stream_count={len(stream_env)} "
            f"startup_s={startup_spawn_seconds + startup_reset_seconds:.3f} "
            f"training_s={training_wall_seconds:.3f} total_s={total_wall_seconds:.3f} "
            f"overall_sps={overall_sps:.2f}"
        )

        writer.flush()
    finally:
        if live_view is not None:
            live_view.close()
        if batch_env is not None:
            batch_env.close()
        writer.close()
