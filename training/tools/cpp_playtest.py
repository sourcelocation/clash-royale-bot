from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.custom.cpp_env import CppClashEnvBatch
from training.custom.env_layout import parse_placement_runtime
from training.custom.knockoff_ppo_support import resolve_torch_device
from training.custom.masked_policy_adapter import MaskedPPOAgent, MaskedPolicyConfig
from training.custom.ppo import branch_sizes_from_action_space
from training.tools.cpp_view_renderer import CARD_NAMES, ArenaRenderer, kv


CARD_COSTS = {
    0: 1,  # Ice Spirit
    1: 4,  # Musketeer
    2: 3,  # Cannon
    3: 4,  # Hog
    4: 1,  # Skeletons
    5: 4,  # Fireball
    6: 2,  # Ice Golem
    7: 2,  # Log
}
OBS_BASE_FEATURES = 7
OBS_HAND_SLOT_COUNT = 4
OBS_CARD_ONEHOT_SIZE = 8
OBS_HAND_SLOT_STRIDE = OBS_CARD_ONEHOT_SIZE * 2


@dataclass(frozen=True)
class CheckpointEntry:
    path: Path
    experiment_name: str
    summary_elo: float | None
    summary: dict[str, Any]


@dataclass
class LoadedBot:
    entry: CheckpointEntry
    model: MaskedPPOAgent
    checkpoint_elo: float | None
    global_step: int
    iteration: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human vs PPO checkpoint playtest GUI.")
    parser.add_argument("--run-dir", type=str, default="training/logs/custom/minimal", help="Checkpoint run root.")
    parser.add_argument("--tick-hz", type=int, default=10, help="Simulation tick rate.")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS cap.")
    parser.add_argument("--seed", type=int, default=1, help="Initial seed.")
    parser.add_argument(
        "--max-sim-seconds",
        type=float,
        default=86400.0,
        help="Match time cap in seconds. Use a large value for near-unlimited time.",
    )
    parser.add_argument("--turbo-multiplier", type=int, default=8, help="Ticks per frame while holding T.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--human-team", type=int, default=1, choices=(0, 1))
    parser.add_argument("--stochastic-bot", action="store_true", help="Sample bot actions instead of greedy.")
    parser.add_argument("--cheat-infinite-elixir", action="store_true")
    return parser.parse_args()


def _parse_hidden_sizes(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, str):
        sizes = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
        if sizes:
            return sizes
    if isinstance(raw, (list, tuple)):
        sizes = tuple(int(x) for x in raw)
        if sizes:
            return sizes
    return (256, 256)


def _scan_checkpoints(run_root: Path) -> list[CheckpointEntry]:
    entries: list[CheckpointEntry] = []
    if not run_root.exists():
        return entries
    exp_dirs = sorted(
        [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("exp_") and p.name[4:].isdigit()],
        reverse=True,
    )
    for exp_dir in exp_dirs:
        summary_path = exp_dir / "timing_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        summary_elo = float(summary["benchmark_rating"]) if "benchmark_rating" in summary else None
        ckpts = sorted(exp_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for ckpt in ckpts:
            entries.append(
                CheckpointEntry(
                    path=ckpt,
                    experiment_name=exp_dir.name,
                    summary_elo=summary_elo,
                    summary=summary,
                )
            )
    return entries


def _decode_hand_card_ids(obs_vector: np.ndarray) -> list[int]:
    hand: list[int] = []
    for i in range(OBS_HAND_SLOT_COUNT):
        base = OBS_BASE_FEATURES + (i * OBS_HAND_SLOT_STRIDE)
        end = base + OBS_CARD_ONEHOT_SIZE
        if end > int(obs_vector.shape[0]):
            hand.append(-1)
            continue
        onehot = np.asarray(obs_vector[base:end], dtype=np.float32)
        max_val = float(onehot.max(initial=0.0))
        if max_val <= 0.5:
            hand.append(-1)
            continue
        card_id = int(np.argmax(onehot))
        hand.append(max(0, min(OBS_CARD_ONEHOT_SIZE - 1, card_id)))
    return hand


def _action_to_region_cell(
    action_x: int,
    action_y: int,
    *,
    grid_w: int,
    grid_h: int,
    region_cols: int,
    region_rows: int,
    region_cell_count: int,
) -> tuple[int, int]:
    region_w = max(1, int(math.ceil(float(grid_w) / float(max(1, region_cols)))))
    region_h = max(1, int(math.ceil(float(grid_h) / float(max(1, region_rows)))))
    rx = min(region_cols - 1, max(0, action_x // region_w))
    ry = min(region_rows - 1, max(0, action_y // region_h))
    local_x = action_x - rx * region_w
    local_y = action_y - ry * region_h
    cell = local_y * region_w + local_x
    cell = min(max(0, cell), max(0, region_cell_count - 1))
    region = ry * region_cols + rx
    return region, cell


class PlaytestApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.human_team = int(args.human_team)
        self.bot_team = 1 - self.human_team
        self.cheat_enabled = bool(args.cheat_infinite_elixir)
        self.pending_action: np.ndarray | None = None
        self.selected_slot = 0
        self.selection_index = 0
        self.status_line = "Select a bot (Enter) and play."
        self.result_text = ""
        self.match_done = False
        self.last_winner = -1
        self.sim_tick = 0
        self.current_fps = 0.0
        self._line_targets: dict[int, tuple[str, int]] = {}

        if str(args.device) == "auto":
            self.device = resolve_torch_device(use_cuda=True, use_mps=True)
        else:
            self.device = torch.device(str(args.device))

        self.env = CppClashEnvBatch(
            num_envs=1,
            tick_hz=int(args.tick_hz),
            max_sim_seconds=float(args.max_sim_seconds),
            seed=int(args.seed),
            num_threads=0,
        )
        self.spec = self.env.spec_data
        self.action_order = list(self.env.action_order)
        self.branch_sizes = branch_sizes_from_action_space(
            self.env.action_space["agent_0"].spaces,
            self.action_order,
        )
        self.action_index = {k: i for i, k in enumerate(self.action_order)}
        self.mask_offsets = self._build_mask_offsets()
        self.wait_action = np.zeros((len(self.action_order),), dtype=np.int32)
        self.wait_action[self.action_index["wait"]] = 1

        placement = parse_placement_runtime(self.spec)
        self.grid_w = int(placement.config.grid_width)
        self.grid_h = int(placement.config.grid_height)
        self.region_cols = int(placement.config.region_cols)
        self.region_rows = int(placement.config.region_rows)
        self.region_cell_count = int(placement.config.region_cell_count)
        self.position_count = int(placement.position_count)

        self.checkpoints = _scan_checkpoints(Path(args.run_dir))
        if not self.checkpoints:
            raise FileNotFoundError(f"No checkpoint files found under {args.run_dir}")

        self.renderer = ArenaRenderer(title="CR PPO Playtest", width=1280, height=760, headless=False)
        self.current_obs = np.zeros((2, int(self.spec["obs_schema"]["vector_size"])), dtype=np.float32)
        self.current_mask = np.zeros((2, int(self.spec["action_mask_size"])), dtype=np.float32)
        self.current_card_masks = np.zeros((2, 8, self.position_count), dtype=np.float32)
        self.debug_state: dict[str, Any] = {"entities": [], "elixir": [0.0, 0.0], "sim_time_s": 0.0}
        self.loaded_bot: LoadedBot | None = None
        self.policy_cfg = self._build_policy_config(sample_obs_dim=int(self.current_obs.shape[1]))

    def _build_mask_offsets(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        offset = 0
        for key, size in zip(self.action_order, self.branch_sizes):
            out[key] = (offset, offset + int(size))
            offset += int(size)
        return out

    def _build_policy_config(self, sample_obs_dim: int) -> MaskedPolicyConfig:
        placement = parse_placement_runtime(self.spec)
        return MaskedPolicyConfig(
            obs_dim=int(sample_obs_dim),
            action_branch_sizes=[int(x) for x in self.branch_sizes],
            action_order=tuple(self.action_order),
            hidden_sizes=(256, 256),
            placement=placement.config,
        )

    def _reset_match(self) -> None:
        options = [
            {
                "team_controllers": ["external", "external"],
                "training_mode": True,
                "ticks_per_step": 1,
                "infinite_elixir_teams": [
                    bool(self.cheat_enabled and self.human_team == 0),
                    bool(self.cheat_enabled and self.human_team == 1),
                ],
            }
        ]
        reset_obs, _ = self.env.reset_many(seeds=[None], options_per_env=options)
        entry = reset_obs[0]
        if entry is None:
            raise RuntimeError("reset_many returned empty observation")
        for team in (0, 1):
            key = f"agent_{team}"
            self.current_obs[team] = np.asarray(entry[key]["vector"], dtype=np.float32)
            self.current_mask[team] = np.asarray(entry[key]["action_mask"], dtype=np.float32)
            self.current_card_masks[team] = np.asarray(entry[key]["position_masks_for_all_cards"], dtype=np.float32)
        self.debug_state = self.env.debug_state_many()[0]
        self.pending_action = None
        self.result_text = ""
        self.match_done = False
        self.last_winner = -1
        self.sim_tick = 0

    def _load_selected_bot(self) -> None:
        entry = self.checkpoints[self.selection_index]
        payload = torch.load(entry.path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Unsupported checkpoint format: {entry.path}")

        args_raw = payload.get("args", {})
        hidden_sizes = _parse_hidden_sizes(args_raw.get("hidden_sizes")) if isinstance(args_raw, dict) else (256, 256)
        self.policy_cfg = MaskedPolicyConfig(
            obs_dim=int(self.current_obs.shape[1]),
            action_branch_sizes=[int(x) for x in self.branch_sizes],
            action_order=tuple(self.action_order),
            hidden_sizes=hidden_sizes,
            placement=parse_placement_runtime(self.spec).config,
        )
        model = MaskedPPOAgent(self.policy_cfg).to(self.device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()

        checkpoint_elo = None
        if "benchmark_rating" in payload:
            checkpoint_elo = float(payload["benchmark_rating"])
        elif entry.summary_elo is not None:
            checkpoint_elo = float(entry.summary_elo)

        self.loaded_bot = LoadedBot(
            entry=entry,
            model=model,
            checkpoint_elo=checkpoint_elo,
            global_step=int(payload.get("global_step", 0)),
            iteration=int(payload.get("iteration", 0)),
        )
        self.status_line = f"Loaded {entry.path.name}"
        self._reset_match()

    def _infer_bot_action(self) -> np.ndarray:
        if self.loaded_bot is None:
            return self.wait_action.copy()
        with torch.no_grad():
            obs_t = torch.as_tensor(self.current_obs[self.bot_team][None, :], dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(self.current_mask[self.bot_team][None, :], dtype=torch.float32, device=self.device)
            card_t = torch.as_tensor(
                self.current_card_masks[self.bot_team][None, :, :],
                dtype=torch.float32,
                device=self.device,
            )
            action_t, _lp, _ent, _val, _rm, _cm = self.loaded_bot.model.get_action_and_value(
                obs_t,
                mask_t,
                card_t,
                deterministic=not bool(self.args.stochastic_bot),
            )
        return action_t[0].detach().cpu().numpy().astype(np.int32, copy=False)

    def _screen_to_action(self, mx: int, my: int) -> tuple[int, int] | None:
        vx, vy, vw, vh = self.renderer.last_viewport
        if vw <= 0 or vh <= 0:
            return None
        if mx < vx or mx >= vx + vw or my < vy or my >= vy + vh:
            return None
        rel_x = (float(mx - vx) / float(vw))
        rel_y = (float(my - vy) / float(vh))
        internal_x = max(0, min(self.grid_w - 1, int(rel_x * self.grid_w)))
        internal_y = max(0, min(self.grid_h - 1, self.grid_h - 1 - int(rel_y * self.grid_h)))
        if self.human_team == 0:
            return (self.grid_w - 1 - internal_x, self.grid_h - 1 - internal_y)
        return (internal_x, internal_y)

    def _queue_human_action(self, mx: int, my: int) -> None:
        if self.match_done:
            return
        hand = _decode_hand_card_ids(self.current_obs[self.human_team])
        if self.selected_slot < 0 or self.selected_slot >= len(hand):
            self.status_line = "Select a card slot first."
            return
        card_id = int(hand[self.selected_slot])
        if card_id < 0:
            self.status_line = "Empty hand slot."
            return
        action_xy = self._screen_to_action(mx, my)
        if action_xy is None:
            self.status_line = "Click inside the arena."
            return
        action_x, action_y = action_xy
        pos_index = action_y * self.grid_w + action_x
        if pos_index < 0 or pos_index >= self.position_count:
            self.status_line = "Invalid position."
            return
        if float(self.current_card_masks[self.human_team, card_id, pos_index]) <= 0.5:
            self.status_line = "Illegal placement for selected card."
            return
        region, cell = _action_to_region_cell(
            action_x,
            action_y,
            grid_w=self.grid_w,
            grid_h=self.grid_h,
            region_cols=self.region_cols,
            region_rows=self.region_rows,
            region_cell_count=self.region_cell_count,
        )
        out = np.zeros((len(self.action_order),), dtype=np.int32)
        out[self.action_index["wait"]] = 0
        out[self.action_index["card_selection"]] = int(card_id)
        out[self.action_index["position_region"]] = int(region)
        out[self.action_index["position_cell"]] = int(cell)
        self.pending_action = out
        self.status_line = f"Queued {CARD_NAMES.get(card_id, str(card_id))} at ({action_x}, {action_y})"

    def _run_sim_steps(self, count: int) -> None:
        if self.loaded_bot is None:
            return
        for _ in range(max(0, int(count))):
            if self.match_done:
                return
            should_decide = (self.sim_tick % max(1, int(self.args.tick_hz))) == 0
            human_action = self.wait_action.copy()
            bot_action = self.wait_action.copy()
            if should_decide:
                if self.pending_action is not None:
                    human_action = self.pending_action
                    self.pending_action = None
                bot_action = self._infer_bot_action()

            actions_per_env = np.zeros((1, 2, len(self.action_order)), dtype=np.int32)
            actions_per_env[:] = self.wait_action
            actions_per_env[0, self.human_team] = human_action
            actions_per_env[0, self.bot_team] = bot_action
            obs, mask, card, _reward, done, trunc, winner, _rt = self.env.step_many_packed(actions_per_env)
            self.current_obs = obs[0]
            self.current_mask = mask[0]
            self.current_card_masks = card[0]
            self.sim_tick += 1
            self.debug_state = self.env.debug_state_many()[0]
            ended = bool(done[0] or trunc[0])
            if ended:
                self.match_done = True
                self.last_winner = int(winner[0])
                if self.last_winner == self.human_team:
                    self.result_text = "Victory"
                elif self.last_winner == self.bot_team:
                    self.result_text = "Loss"
                else:
                    self.result_text = "Draw"
                self.status_line = "Press R to reset match."
                return

    def _build_hud(self) -> list[tuple[str, tuple[str, int] | None]]:
        rows: list[tuple[str, tuple[str, int] | None]] = []
        bot = self.loaded_bot
        sim_time = float(self.debug_state.get("sim_time_s", 0.0))
        elixir = self.debug_state.get("elixir", [0.0, 0.0])
        human_elixir = float(elixir[self.human_team]) if len(elixir) > self.human_team else 0.0
        hand = _decode_hand_card_ids(self.current_obs[self.human_team])
        card_mask_off = self.mask_offsets["card_selection"][0]
        playable = [
            bool(float(self.current_mask[self.human_team, card_mask_off + max(0, card_id)]) > 0.5) if card_id >= 0 else False
            for card_id in hand
        ]

        rows.extend(
            [
                ("[Checkpoint]", None),
                (kv("Loaded", bot.entry.path.name if bot is not None else "(none)", 13), None),
                (kv("Experiment", bot.entry.experiment_name if bot is not None else "-", 13), None),
                (kv("ELO", f"{float(bot.checkpoint_elo):.1f}" if (bot and bot.checkpoint_elo is not None) else "-", 13), None),
                (kv("Step", str(bot.global_step if bot else "-"), 13), None),
                (kv("Iteration", str(bot.iteration if bot else "-"), 13), None),
                ("", None),
                ("[Match]", None),
                (kv("FPS", f"{self.current_fps:.1f}", 13), None),
                (kv("Time", f"{sim_time:.1f}s", 13), None),
                (kv("Elixir", f"{human_elixir:.2f}", 13), None),
                (kv("Cheat", "ON" if self.cheat_enabled else "OFF", 13), None),
                (kv("Result", self.result_text or "-", 13), None),
                ("", None),
                ("[Hand] click line or press 1..4", None),
            ]
        )
        for slot in range(4):
            card_id = hand[slot] if slot < len(hand) else -1
            selected = ">" if slot == self.selected_slot else " "
            name = CARD_NAMES.get(card_id, "Empty")
            cost = CARD_COSTS.get(card_id, 0)
            play = "play" if (slot < len(playable) and playable[slot]) else "hold"
            rows.append((f"{selected} [{slot + 1}] {name:<10} cost={cost} {play}", ("hand", slot)))

        rows.append(("", None))
        rows.append(("[Bots] click line / Up-Down / Enter", None))
        visible = 9
        start = max(0, self.selection_index - (visible // 2))
        start = min(start, max(0, len(self.checkpoints) - visible))
        end = min(len(self.checkpoints), start + visible)
        for i in range(start, end):
            ck = self.checkpoints[i]
            marker = ">" if i == self.selection_index else " "
            rows.append((f"{marker} {ck.experiment_name}/{ck.path.name}", ("bot", i)))

        rows.extend(
            [
                ("", None),
                (self.status_line, None),
                ("Controls: T hold speed, C cheat toggle+reset, R reset, ESC quit", None),
            ]
        )
        return rows

    def _update_line_targets(self, rows: list[tuple[str, tuple[str, int] | None]]) -> None:
        self._line_targets.clear()
        width, height = self.renderer.screen.get_size()
        hud_rect, _ = self.renderer._layout_panels(width, height)
        hx, hy, hw, _hh = hud_rect
        y = hy + 10
        for line, tag in rows:
            if line == "":
                y += 6
                continue
            if tag is not None:
                self._line_targets[y] = tag
            y += 18
        self._hud_bounds = (hx, hy, hw, height)

    def _handle_click(self, mx: int, my: int) -> None:
        hx, hy, hw, hh = self._hud_bounds
        if hx <= mx <= hx + hw and hy <= my <= hy + hh:
            nearest = None
            nearest_dist = 1e9
            for y, tag in self._line_targets.items():
                d = abs(my - y)
                if d < nearest_dist and d <= 10:
                    nearest_dist = d
                    nearest = tag
            if nearest is not None:
                kind, idx = nearest
                if kind == "hand":
                    self.selected_slot = int(idx)
                    self.status_line = f"Selected card slot {self.selected_slot + 1}"
                    return
                if kind == "bot":
                    self.selection_index = int(idx)
                    try:
                        self._load_selected_bot()
                    except Exception as exc:
                        self.status_line = f"Failed to load checkpoint: {exc}"
                    return
            return
        self._queue_human_action(mx, my)

    def run(self) -> None:
        try:
            self._load_selected_bot()
        except Exception as exc:
            self.status_line = f"Initial load failed: {exc}"

        running = True
        self._hud_bounds = (0, 0, 1, 1)
        while running and not self.renderer.closed:
            events = self.renderer.poll_events(esc_quit=False)
            if events.get("quit") or events.get("esc"):
                break

            for key in events.get("key_downs", []):
                if key == self.renderer.pygame.K_r:
                    self._reset_match()
                elif key == self.renderer.pygame.K_c:
                    self.cheat_enabled = not self.cheat_enabled
                    self.status_line = "Cheat mode toggled. Match reset."
                    self._reset_match()
                elif key in (self.renderer.pygame.K_1, self.renderer.pygame.K_2, self.renderer.pygame.K_3, self.renderer.pygame.K_4):
                    self.selected_slot = int(key - self.renderer.pygame.K_1)
                elif key == self.renderer.pygame.K_UP:
                    self.selection_index = max(0, self.selection_index - 1)
                elif key == self.renderer.pygame.K_DOWN:
                    self.selection_index = min(len(self.checkpoints) - 1, self.selection_index + 1)
                elif key in (self.renderer.pygame.K_RETURN, self.renderer.pygame.K_KP_ENTER):
                    try:
                        self._load_selected_bot()
                    except Exception as exc:
                        self.status_line = f"Failed to load checkpoint: {exc}"

            for mx, my in events.get("mouse_left_clicks", []):
                self._handle_click(int(mx), int(my))

            turbo = bool(self.renderer.pygame.key.get_pressed()[self.renderer.pygame.K_t])
            steps = int(self.args.turbo_multiplier) if turbo else 1
            self._run_sim_steps(steps)

            hud_rows = self._build_hud()
            hud_lines = [line for line, _ in hud_rows]
            frame = self.renderer.draw(
                self.debug_state,
                hud_lines,
                fps_limit=max(1, int(self.args.fps)),
            )
            self.current_fps = float(frame.get("fps", 0.0))
            self._update_line_targets(hud_rows)
        self.close()

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.renderer.close()


def main() -> None:
    args = _parse_args()
    app = PlaytestApp(args)
    app.run()


if __name__ == "__main__":
    main()
