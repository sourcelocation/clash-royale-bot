"""Minimal packed-env PPO with a GPU-resident Tier-A opponent pool."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .cpp_env import CppClashEnvBatch
from .env_layout import as_card_position_masks, parse_placement_runtime
from .knockoff_ppo_support import resolve_torch_device
from .masked_policy_adapter import MaskedPPOAgent, MaskedPolicyConfig
from .minimal_ppo_args import Args
from .ppo import branch_sizes_from_action_space


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
    if not sizes:
        raise ValueError("hidden_sizes must not be empty")
    return sizes


def _next_experiment_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in run_root.iterdir():
        if child.is_dir() and child.name.startswith("exp_") and child.name[4:].isdigit():
            max_index = max(max_index, int(child.name[4:]))
    return run_root / f"exp_{max_index + 1:04d}"


class _TrainingVisualizer:
    def __init__(self, fps: int):
        from training.tools.cpp_view_renderer import ArenaRenderer, kv

        self.fps = max(1, int(fps))
        self.env_index = 0
        self._kv = kv
        self.renderer = ArenaRenderer(title="minimal ppo live view", width=1100, height=620, headless=False)
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
        env_codes = [int(x) for x in metrics.get("env_opponent_codes", [])]
        env_count = max(1, int(metrics.get("num_envs", max(1, len(env_codes)))))
        grid_cols = max(1, int(math.ceil(math.sqrt(env_count))))
        steps_to_update = int(metrics.get("steps_to_update", 0))
        rollout_step = int(metrics.get("rollout_step", 0))
        total_rollout_steps = max(1, int(metrics.get("rollout_total_steps", 1)))
        progress_pct = 100.0 * float(metrics.get("progress_frac", 0.0))
        if self.collapsed:
            lines = [
                "[MINIMAL TRAINING VIEW]",
                "Collapsed mode: training running in background",
                self._kv("Global step", str(int(metrics.get("global_step", 0)))),
                self._kv("Iteration", str(int(metrics.get("iteration", 0)))),
                self._kv("SPS", str(int(metrics.get("sps", 0)))),
                self._kv("Rollout step", f"{rollout_step}/{total_rollout_steps}"),
                self._kv("Steps to update", str(steps_to_update)),
                self._kv("Progress", f"{progress_pct:.1f}%"),
                self._kv("Env", str(self.env_index)),
                "",
                "Press Esc to restore full preview",
                "Left/Right: switch env",
                "Close window to disable viewer",
            ]
            self.renderer.draw(
                {"entities": [], "elixir": [0.0, 0.0]},
                lines,
                fps_limit=5,
                hud_grid={
                    "title": "Opponent Per Env",
                    "cells": env_codes,
                    "cols": grid_cols,
                    "cell_size": 14,
                    "palette": {
                        0: (70, 140, 230),   # latest-latest
                        1: (230, 145, 60),   # latest-recent
                        2: (80, 180, 110),   # latest-anchor
                    },
                    "legend": [
                        {"label": "Latest vs Latest", "color": (70, 140, 230)},
                        {"label": "Latest vs Recent", "color": (230, 145, 60)},
                        {"label": "Latest vs Anchor", "color": (80, 180, 110)},
                    ],
                },
            )
            return True

        lines = [
            "[MINIMAL TRAINING]",
            self._kv("Env index", str(self.env_index)),
            self._kv("Global step", str(int(metrics.get("global_step", 0)))),
            self._kv("Iteration", str(int(metrics.get("iteration", 0)))),
            self._kv("SPS", str(int(metrics.get("sps", 0)))),
            self._kv("Rollout step", f"{rollout_step}/{total_rollout_steps}"),
            self._kv("Steps to update", str(steps_to_update)),
            self._kv("Progress", f"{progress_pct:.1f}%"),
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
        self.renderer.draw(
            state,
            lines,
            fps_limit=self.fps,
            hud_grid={
                "title": "Opponent Per Env",
                "cells": env_codes,
                "cols": grid_cols,
                "cell_size": 14,
                "palette": {
                    0: (70, 140, 230),   # latest-latest
                    1: (230, 145, 60),   # latest-recent
                    2: (80, 180, 110),   # latest-anchor
                },
                "legend": [
                    {"label": "Latest vs Latest", "color": (70, 140, 230)},
                    {"label": "Latest vs Recent", "color": (230, 145, 60)},
                    {"label": "Latest vs Anchor", "color": (80, 180, 110)},
                ],
            },
        )
        return True

    def close(self) -> None:
        self.renderer.close()


@dataclass
class _PolicySnapshot:
    step: int
    state_dict: dict[str, torch.Tensor]


class _GpuOpponentPool:
    def __init__(self, *, args: Args, policy_cfg: MaskedPolicyConfig, device: torch.device):
        self.enabled = bool(args.pool_enabled)
        self.device = device
        self.policy_cfg = policy_cfg
        self.recent_capacity = max(1, int(args.pool_recent_capacity))
        self.anchor_capacity = max(0, int(args.pool_anchor_capacity))
        self.active_recent_size = max(0, int(args.pool_active_recent_size))
        self.active_anchor_size = max(0, int(args.pool_active_anchor_size))
        self.promote_every = max(1, int(args.pool_promote_every))
        self.refresh_every = max(1, int(args.pool_refresh_every))
        self.anchor_every = max(1, int(args.pool_anchor_every))
        self.latest_latest_prob = max(0.0, float(args.pool_latest_latest_prob))
        self.latest_recent_prob = max(0.0, float(args.pool_latest_recent_prob))
        self.latest_anchor_prob = max(0.0, float(args.pool_latest_anchor_prob))
        self.opponent_deterministic = bool(args.pool_deterministic)

        self.rng = np.random.default_rng(int(args.seed) + 7919)
        self.recent: list[_PolicySnapshot] = []
        self.anchors: list[_PolicySnapshot] = []
        self.active_agents: dict[int, MaskedPPOAgent] = {}
        self.active_recent_ids: list[int] = []
        self.active_anchor_ids: list[int] = []
        self._next_policy_id = 1

    def _clone_state(self, agent: MaskedPPOAgent) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}

    def _build_frozen_agent(self, snapshot: _PolicySnapshot) -> MaskedPPOAgent:
        opp = MaskedPPOAgent(self.policy_cfg).to(self.device)
        opp.load_state_dict(snapshot.state_dict)
        opp.eval()
        return opp

    def bootstrap(self, latest_agent: MaskedPPOAgent) -> None:
        if not self.enabled:
            return
        self.promote(latest_agent, step=0)
        self.refresh_active()

    def promote(self, latest_agent: MaskedPPOAgent, *, step: int) -> None:
        if not self.enabled:
            return
        snap = _PolicySnapshot(step=int(step), state_dict=self._clone_state(latest_agent))
        self.recent.append(snap)
        if len(self.recent) > self.recent_capacity:
            self.recent.pop(0)
        if self.anchor_capacity > 0 and (step % self.anchor_every == 0):
            self.anchors.append(snap)
            if len(self.anchors) > self.anchor_capacity:
                self.anchors.pop(0)

    def refresh_active(self) -> None:
        if not self.enabled:
            return
        self.active_agents = {}
        self.active_recent_ids = []
        self.active_anchor_ids = []
        self._next_policy_id = 1

        recent_count = min(self.active_recent_size, len(self.recent))
        anchor_count = min(self.active_anchor_size, len(self.anchors))
        if recent_count > 0:
            idxs = self.rng.choice(len(self.recent), size=recent_count, replace=False)
            for idx in np.atleast_1d(idxs):
                pid = self._next_policy_id
                self._next_policy_id += 1
                self.active_agents[pid] = self._build_frozen_agent(self.recent[int(idx)])
                self.active_recent_ids.append(pid)
        if anchor_count > 0:
            idxs = self.rng.choice(len(self.anchors), size=anchor_count, replace=False)
            for idx in np.atleast_1d(idxs):
                pid = self._next_policy_id
                self._next_policy_id += 1
                self.active_agents[pid] = self._build_frozen_agent(self.anchors[int(idx)])
                self.active_anchor_ids.append(pid)

    def should_refresh(self, iteration: int) -> bool:
        return self.enabled and (iteration % self.refresh_every == 0)

    def should_promote(self, iteration: int) -> bool:
        return self.enabled and (iteration % self.promote_every == 0)

    def choose_match(self) -> tuple[str, int | None]:
        if not self.enabled:
            return "latest_latest", None

        total = self.latest_latest_prob + self.latest_recent_prob + self.latest_anchor_prob
        if total <= 0.0:
            return "latest_latest", None
        p_ll = self.latest_latest_prob / total
        p_lr = self.latest_recent_prob / total

        r = float(self.rng.random())
        if r < p_ll:
            return "latest_latest", None
        if r < p_ll + p_lr and self.active_recent_ids:
            pid = int(self.active_recent_ids[int(self.rng.integers(len(self.active_recent_ids)))])
            return "latest_recent", pid
        if self.active_anchor_ids:
            pid = int(self.active_anchor_ids[int(self.rng.integers(len(self.active_anchor_ids)))])
            return "latest_anchor", pid
        if self.active_recent_ids:
            pid = int(self.active_recent_ids[int(self.rng.integers(len(self.active_recent_ids)))])
            return "latest_recent", pid
        return "latest_latest", None


def run_training(args: Args) -> None:
    if args.num_envs <= 0 or args.num_steps <= 0 or args.num_minibatches <= 0:
        raise ValueError("num_envs/num_steps/num_minibatches must be > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    device = resolve_torch_device(use_cuda=bool(args.cuda), use_mps=bool(args.mps))
    run_root = Path(args.run_dir)
    experiment_dir = _next_experiment_dir(run_root)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    tb_root = Path(args.tb_dir) if str(args.tb_dir).strip() else (run_root / "tensorboard")
    tb_dir = tb_root / experiment_dir.name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_text("run/paths", f"experiment_dir={experiment_dir}\ntensorboard_dir={tb_dir}\n")
    print(f"[MIN] experiment={experiment_dir.name} run_dir={experiment_dir} tb_dir={tb_dir}")

    env = CppClashEnvBatch(
        num_envs=int(args.num_envs),
        tick_hz=int(args.cpp_tick_hz),
        max_sim_seconds=float(args.cpp_max_sim_seconds),
        seed=int(args.seed),
        num_threads=int(args.cpp_num_threads),
    )
    live_view: _TrainingVisualizer | None = None
    live_view_failed = False
    try:
        if bool(args.visualize):
            try:
                live_view = _TrainingVisualizer(fps=int(args.visualize_fps))
                print(f"[MIN] live viewer enabled fps={int(args.visualize_fps)}")
            except Exception as exc:
                live_view_failed = True
                print(f"[MIN] live viewer disabled: {exc}")

        action_order = env.action_order
        agent_count = len(env.agent_keys)
        if agent_count != 2:
            raise ValueError(f"Minimal pool trainer currently supports exactly 2 agents, got {agent_count}")

        reset_options = [
            {
                "team_controllers": ["external" for _ in range(agent_count)],
                "training_mode": True,
                "ticks_per_step": int(args.cpp_tick_hz),
            }
            for _ in range(args.num_envs)
        ]
        reset_obs, _ = env.reset_many(
            seeds=[int(args.seed) + i for i in range(args.num_envs)],
            options_per_env=reset_options,
        )
        sample = reset_obs[0]
        if sample is None:
            raise RuntimeError("reset returned empty observation for env 0")

        sample_agent = "agent_0"
        obs_dim = int(len(sample[sample_agent]["vector"]))
        mask_dim = int(len(sample[sample_agent]["action_mask"]))
        placement_runtime = parse_placement_runtime(env.spec_data)
        cards_count, position_count = as_card_position_masks(
            sample[sample_agent],
            expected_position_count=placement_runtime.position_count,
        ).shape
        branch_sizes = branch_sizes_from_action_space(env.action_space[sample_agent].spaces, action_order)
        num_branches = len(branch_sizes)

        policy_cfg = MaskedPolicyConfig(
            obs_dim=obs_dim,
            action_branch_sizes=branch_sizes,
            action_order=tuple(action_order),
            hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
            placement=placement_runtime.config,
        )
        agent = MaskedPPOAgent(policy_cfg).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=float(args.learning_rate), eps=1e-5)

        pool = _GpuOpponentPool(args=args, policy_cfg=policy_cfg, device=device)
        pool.bootstrap(agent)

        current_obs = np.zeros((args.num_envs, agent_count, obs_dim), dtype=np.float32)
        current_mask = np.zeros((args.num_envs, agent_count, mask_dim), dtype=np.float32)
        current_card = np.zeros((args.num_envs, agent_count, cards_count, position_count), dtype=np.float32)

        def write_reset(env_idx: int, obs_entry: dict) -> None:
            for team_idx in range(agent_count):
                key = f"agent_{team_idx}"
                current_obs[env_idx, team_idx] = np.asarray(obs_entry[key]["vector"], dtype=np.float32)
                current_mask[env_idx, team_idx] = np.asarray(obs_entry[key]["action_mask"], dtype=np.float32)
                current_card[env_idx, team_idx] = as_card_position_masks(
                    obs_entry[key],
                    expected_position_count=placement_runtime.position_count,
                )

        for env_idx in range(args.num_envs):
            obs_entry = reset_obs[env_idx]
            if obs_entry is None:
                raise RuntimeError(f"reset returned empty observation for env {env_idx}")
            write_reset(env_idx, obs_entry)

        stream_env = np.repeat(np.arange(args.num_envs, dtype=np.int64), agent_count)
        stream_team = np.tile(np.arange(agent_count, dtype=np.int64), args.num_envs)
        stream_count = int(stream_env.shape[0])
        batch_size = int(stream_count * args.num_steps)
        minibatch_size = max(1, batch_size // int(args.num_minibatches))
        num_iterations = max(1, int(math.ceil(float(args.total_timesteps) / float(batch_size))))
        stream_index = np.arange(stream_count, dtype=np.int64).reshape(args.num_envs, agent_count)

        wait_idx = action_order.index("wait") if "wait" in action_order else None
        wait_action = np.zeros((num_branches,), dtype=np.int32)
        if wait_idx is not None:
            wait_action[wait_idx] = 1

        stream_policy_id = np.zeros((stream_count,), dtype=np.int64)
        stream_trainable = np.ones((stream_count,), dtype=np.float32)
        env_match_tag: list[str] = ["latest_latest" for _ in range(args.num_envs)]

        assign_rng = np.random.default_rng(int(args.seed) + 17)

        def assign_env(env_idx: int) -> None:
            tag, opp_pid = pool.choose_match()
            env_match_tag[env_idx] = tag
            idx0 = int(stream_index[env_idx, 0])
            idx1 = int(stream_index[env_idx, 1])

            if tag == "latest_latest" or opp_pid is None:
                stream_policy_id[idx0] = 0
                stream_policy_id[idx1] = 0
                stream_trainable[idx0] = 1.0
                stream_trainable[idx1] = 1.0
                return

            latest_team = int(assign_rng.integers(0, 2))
            opp_team = 1 - latest_team
            latest_idx = int(stream_index[env_idx, latest_team])
            opp_idx = int(stream_index[env_idx, opp_team])
            stream_policy_id[latest_idx] = 0
            stream_policy_id[opp_idx] = int(opp_pid)
            stream_trainable[latest_idx] = 1.0
            stream_trainable[opp_idx] = 0.0

        for env_idx in range(args.num_envs):
            assign_env(env_idx)

        obs = torch.zeros((args.num_steps, stream_count, obs_dim), dtype=torch.float32, device=device)
        masks = torch.zeros((args.num_steps, stream_count, mask_dim), dtype=torch.float32, device=device)
        card_masks = torch.zeros(
            (args.num_steps, stream_count, cards_count, position_count), dtype=torch.float32, device=device
        )
        region_masks = torch.zeros(
            (args.num_steps, stream_count, branch_sizes[action_order.index("position_region")]),
            dtype=torch.float32,
            device=device,
        )
        cell_masks = torch.zeros(
            (args.num_steps, stream_count, branch_sizes[action_order.index("position_cell")]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.zeros((args.num_steps, stream_count, num_branches), dtype=torch.long, device=device)
        logprobs = torch.zeros((args.num_steps, stream_count), dtype=torch.float32, device=device)
        rewards = torch.zeros((args.num_steps, stream_count), dtype=torch.float32, device=device)
        dones = torch.zeros((args.num_steps, stream_count), dtype=torch.float32, device=device)
        values = torch.zeros((args.num_steps, stream_count), dtype=torch.float32, device=device)
        train_mask = torch.zeros((args.num_steps, stream_count), dtype=torch.float32, device=device)

        def stream_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.as_tensor(current_obs[stream_env, stream_team], dtype=torch.float32, device=device),
                torch.as_tensor(current_mask[stream_env, stream_team], dtype=torch.float32, device=device),
                torch.as_tensor(current_card[stream_env, stream_team], dtype=torch.float32, device=device),
            )

        next_obs, next_mask, next_card = stream_tensors()
        next_done = torch.zeros((stream_count,), dtype=torch.float32, device=device)

        global_step = 0
        train_start = time.perf_counter()
        pending_pool_refresh = False
        done_since_refresh = np.zeros((args.num_envs,), dtype=bool)

        for iteration in range(1, num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / max(1, num_iterations)
                optimizer.param_groups[0]["lr"] = frac * float(args.learning_rate)

            if pool.should_refresh(iteration):
                pending_pool_refresh = True
                done_since_refresh[:] = False

            rollout_start = time.perf_counter()
            for step in range(args.num_steps):
                global_step += stream_count
                obs[step] = next_obs
                masks[step] = next_mask
                card_masks[step] = next_card
                dones[step] = next_done
                train_mask[step] = torch.as_tensor(stream_trainable, dtype=torch.float32, device=device)

                actions_stream = torch.zeros((stream_count, num_branches), dtype=torch.long, device=device)
                logprobs_stream = torch.zeros((stream_count,), dtype=torch.float32, device=device)
                values_stream = torch.zeros((stream_count,), dtype=torch.float32, device=device)
                regions_stream = torch.zeros((stream_count, region_masks.shape[-1]), dtype=torch.float32, device=device)
                cells_stream = torch.zeros((stream_count, cell_masks.shape[-1]), dtype=torch.float32, device=device)

                for pid in np.unique(stream_policy_id):
                    idx_np = np.flatnonzero(stream_policy_id == int(pid))
                    if idx_np.size == 0:
                        continue
                    idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                    policy = agent if int(pid) == 0 else pool.active_agents.get(int(pid))
                    if policy is None:
                        policy = agent
                    with torch.no_grad():
                        act, lp, _ent, val, rm, cm = policy.get_action_and_value(
                            next_obs[idx_t],
                            next_mask[idx_t],
                            next_card[idx_t],
                            deterministic=bool(pool.opponent_deterministic and int(pid) != 0),
                        )
                    actions_stream[idx_t] = act
                    logprobs_stream[idx_t] = lp
                    values_stream[idx_t] = val.flatten()
                    regions_stream[idx_t] = rm
                    cells_stream[idx_t] = cm

                actions[step] = actions_stream
                logprobs[step] = logprobs_stream
                values[step] = values_stream
                region_masks[step] = regions_stream
                cell_masks[step] = cells_stream

                action_np = actions_stream.detach().cpu().numpy().astype(np.int32, copy=False)
                actions_per_env = np.empty((args.num_envs, agent_count, num_branches), dtype=np.int32)
                actions_per_env[:] = wait_action
                actions_per_env[stream_env, stream_team] = action_np

                packed_obs, packed_mask, packed_card, packed_reward, packed_done, packed_trunc, _winner, _rt = (
                    env.step_many_packed(actions_per_env)
                )
                current_obs = packed_obs
                current_mask = packed_mask
                current_card = packed_card

                done_env = np.logical_or(packed_done, packed_trunc)
                rewards[step] = torch.as_tensor(
                    packed_reward[stream_env, stream_team], dtype=torch.float32, device=device
                )
                next_done = torch.as_tensor(done_env[stream_env].astype(np.float32), dtype=torch.float32, device=device)

                done_ids = np.flatnonzero(done_env)
                if pending_pool_refresh and done_ids.size > 0:
                    done_since_refresh[done_ids] = True

                if pending_pool_refresh and bool(done_since_refresh.all()):
                    pool.refresh_active()
                    pending_pool_refresh = False
                    done_since_refresh[:] = False

                if done_ids.size > 0:
                    options: list[dict | None] = [None for _ in range(args.num_envs)]
                    for env_idx in done_ids:
                        options[int(env_idx)] = {
                            "team_controllers": ["external" for _ in range(agent_count)],
                            "training_mode": True,
                            "ticks_per_step": int(args.cpp_tick_hz),
                        }
                    reset_many_obs, _ = env.reset_many(
                        seeds=[None for _ in range(args.num_envs)],
                        options_per_env=options,
                    )
                    for env_idx in done_ids:
                        env_i = int(env_idx)
                        obs_entry = reset_many_obs[env_i]
                        if obs_entry is None:
                            raise RuntimeError(f"reset did not return observation for env {env_i}")
                        write_reset(env_i, obs_entry)
                        assign_env(env_i)

                if live_view is not None and not live_view_failed:
                    try:
                        env_codes = [
                            0 if tag == "latest_latest" else 1 if tag == "latest_recent" else 2
                            for tag in env_match_tag
                        ]
                        rollout_step = int(step + 1)
                        keep_open = live_view.update(
                            env.debug_state_many(),
                            {
                                "global_step": global_step,
                                "iteration": iteration,
                                "sps": int(global_step / max(1e-9, time.perf_counter() - train_start)),
                                "rollout_step": rollout_step,
                                "rollout_total_steps": int(args.num_steps),
                                "steps_to_update": max(0, int(args.num_steps) - rollout_step),
                                "progress_frac": min(1.0, float(global_step) / float(max(1, args.total_timesteps))),
                                "env_opponent_codes": env_codes,
                                "num_envs": int(args.num_envs),
                            },
                        )
                        if not keep_open:
                            live_view.close()
                            live_view = None
                    except Exception as exc:
                        live_view_failed = True
                        live_view.close()
                        live_view = None
                        print(f"[MIN] live viewer disabled after runtime error: {exc}")

                next_obs, next_mask, next_card = stream_tensors()
            rollout_s = time.perf_counter() - rollout_start

            update_start = time.perf_counter()
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0.0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + float(args.gamma) * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + float(args.gamma) * float(args.gae_lambda) * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            b_obs = obs.reshape((-1, obs_dim))
            b_masks = masks.reshape((-1, mask_dim))
            b_card = card_masks.reshape((-1, cards_count, position_count))
            b_region = region_masks.reshape((-1, region_masks.shape[-1]))
            b_cell = cell_masks.reshape((-1, cell_masks.shape[-1]))
            b_actions = actions.reshape((-1, num_branches))
            b_logprobs = logprobs.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            b_val = values.reshape(-1)
            b_train = train_mask.reshape(-1)

            active = torch.nonzero(b_train > 0.5, as_tuple=False).reshape(-1)
            if active.numel() == 0:
                continue
            active_size = int(active.numel())
            mb_size = max(1, active_size // int(args.num_minibatches))

            approx_kl = torch.tensor(0.0, device=device)
            clipfrac_sum = torch.tensor(0.0, device=device)
            clipfrac_count = 0
            for _ in range(args.update_epochs):
                perm = active[torch.randperm(active_size, device=device)]
                for start in range(0, active_size, mb_size):
                    mb = perm[start : start + mb_size]
                    _, newlogprob, entropy, newvalue, _rm, _cm = agent.get_action_and_value(
                        b_obs[mb],
                        b_masks[mb],
                        b_card[mb],
                        b_actions[mb],
                        b_region[mb],
                        b_cell[mb],
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipfrac_sum = clipfrac_sum + ((ratio - 1.0).abs() > float(args.clip_coef)).float().mean()
                        clipfrac_count += 1

                    mb_adv = b_adv[mb]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / mb_adv.std(unbiased=False).clamp_min(1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - float(args.clip_coef), 1.0 + float(args.clip_coef))
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_ret[mb]) ** 2
                        v_clipped = b_val[mb] + torch.clamp(
                            newvalue - b_val[mb], -float(args.clip_coef), float(args.clip_coef)
                        )
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - float(args.ent_coef) * entropy_loss + float(args.vf_coef) * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), float(args.max_grad_norm))
                    optimizer.step()

                if args.target_kl is not None and float(approx_kl.item()) > float(args.target_kl):
                    break

            if pool.should_promote(iteration):
                pool.promote(agent, step=global_step)

            update_s = time.perf_counter() - update_start
            should_log = int(args.log_every) <= 0 or (iteration % int(args.log_every) == 0)
            if should_log:
                elapsed = max(1e-9, time.perf_counter() - train_start)
                sps = int(global_step / elapsed)
                clipfrac = float((clipfrac_sum / max(1, clipfrac_count)).item())
                ll = int(sum(1 for t in env_match_tag if t == "latest_latest"))
                lr = int(sum(1 for t in env_match_tag if t == "latest_recent"))
                la = int(sum(1 for t in env_match_tag if t == "latest_anchor"))
                iter_s = rollout_s + update_s
                env_count = max(1, ll + lr + la)
                trainable_ratio = float(active_size) / float(max(1, int(b_train.numel())))
                pool_active_total = int(len(pool.active_recent_ids) + len(pool.active_anchor_ids))
                print(
                    f"[MIN] iter={iteration}/{num_iterations} step={global_step} sps={sps} "
                    f"rollout_s={rollout_s:.3f} update_s={update_s:.3f} "
                    f"pg={float(pg_loss.item()):.4f} v={float(v_loss.item()):.4f} kl={float(approx_kl.item()):.5f} "
                    f"cf={clipfrac:.4f} mix(ll/lr/la)={ll}/{lr}/{la} active_pool(r/a)={len(pool.active_recent_ids)}/{len(pool.active_anchor_ids)}"
                )
                writer.add_scalar("charts/sps", sps, global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("timing/iteration_seconds", iter_s, global_step)
                writer.add_scalar("timing/rollout_seconds", rollout_s, global_step)
                writer.add_scalar("timing/update_seconds", update_s, global_step)
                writer.add_scalar("losses/policy_loss", float(pg_loss.item()), global_step)
                writer.add_scalar("losses/value_loss", float(v_loss.item()), global_step)
                writer.add_scalar("losses/entropy", float(entropy_loss.item()), global_step)
                writer.add_scalar("losses/approx_kl", float(approx_kl.item()), global_step)
                writer.add_scalar("losses/clipfrac", clipfrac, global_step)
                writer.add_scalar("pool/latest_latest_ratio", float(ll) / float(env_count), global_step)
                writer.add_scalar("pool/latest_recent_ratio", float(lr) / float(env_count), global_step)
                writer.add_scalar("pool/latest_anchor_ratio", float(la) / float(env_count), global_step)
                writer.add_scalar("pool/trainable_sample_ratio", trainable_ratio, global_step)
                writer.add_scalar("pool/active_policy_count", pool_active_total, global_step)

            if int(args.ckpt_every) > 0 and iteration % int(args.ckpt_every) == 0:
                ckpt_path = experiment_dir / f"ckpt_iter_{iteration:06d}.pt"
                torch.save(
                    {
                        "args": vars(args),
                        "iteration": int(iteration),
                        "global_step": int(global_step),
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )

        total_s = max(1e-9, time.perf_counter() - train_start)
        summary = {
            "experiment_dir": str(experiment_dir),
            "tensorboard_dir": str(tb_dir),
            "global_step": int(global_step),
            "num_iterations": int(num_iterations),
            "overall_sps": float(global_step / total_s),
            "device": str(device),
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "batch_size": int(batch_size),
            "pool_enabled": bool(args.pool_enabled),
            "active_recent": int(len(pool.active_recent_ids)),
            "active_anchor": int(len(pool.active_anchor_ids)),
        }
        (experiment_dir / "timing_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        writer.add_scalar("timing/overall_sps", summary["overall_sps"], int(global_step))
        writer.flush()
        print(
            f"[MIN] done exp={experiment_dir.name} step={global_step} "
            f"sps={summary['overall_sps']:.2f} device={device}"
        )
    finally:
        if live_view is not None:
            live_view.close()
        env.close()
        writer.close()
