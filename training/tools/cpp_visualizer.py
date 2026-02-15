from __future__ import annotations

import argparse
import random
import time
from typing import Any

from training.custom.cpp_env import CppClashEnv
from training.tools.cpp_view_renderer import ArenaRenderer, kv

VIEW_TICK_HZ = 10


def _split_mask(mask: list[float], branch_sizes: dict[str, int], action_order: list[str]) -> dict[str, list[float]]:
    values = list(mask)
    out: dict[str, list[float]] = {}
    offset = 0
    for key in action_order:
        size = int(branch_sizes[key])
        out[key] = values[offset : offset + size]
        offset += size
    return out


def _sample_action(mask: list[float], action_order: list[str], branch_sizes: dict[str, int], rng: random.Random) -> dict[str, int]:
    branches = _split_mask(mask, branch_sizes, action_order)
    action: dict[str, int] = {}
    for key in action_order:
        valid = [i for i, v in enumerate(branches[key]) if float(v) > 0.0]
        action[key] = rng.choice(valid) if valid else 0
    return action


def run_visualizer(
    *,
    max_sim_seconds: float,
    seed: int,
    fps: int,
    decision_hz: float,
    limit_steps_per_second: float | None = None,
    best_effort: bool = True,
    max_frames: int | None = None,
    headless: bool = False,
) -> None:
    renderer = ArenaRenderer(title="knockoff_cr_cpp visualizer", width=1080, height=960, headless=headless)
    env = CppClashEnv(
        env_id=0,
        tick_hz=VIEW_TICK_HZ,
        max_sim_seconds=max_sim_seconds,
        seed=seed,
    )

    try:
        obs, _ = env.reset(seed=seed, options={"ticks_per_step": 1})
        state = env.debug_state()
        rng = random.Random(seed)
        action_order = list(env.action_order)
        branch_sizes = {k: int(env.action_space[env.agent_keys[0]][k].n) for k in action_order}
        decision_interval_ticks = max(1, int(round(VIEW_TICK_HZ / max(1e-6, float(decision_hz)))))
        sim_limit_steps_per_second = float(limit_steps_per_second) if limit_steps_per_second is not None else None
        if sim_limit_steps_per_second is not None:
            sim_limit_steps_per_second = max(1e-6, sim_limit_steps_per_second)
        unthrottled_steps = sim_limit_steps_per_second is None
        sim_accum_steps = 0.0
        max_steps_per_frame = 5000
        sim_time_budget_fraction = 0.96
        sim_steps_window_count = 0
        sim_steps_window_s = 0.0
        sim_steps_actual_sps = 0.0
        sim_steps_dropped_window_count = 0
        sim_steps_dropped_sps = 0.0
        paused = False
        frame_count = 0
        sim_tick_count = 0
        reset_seed = seed
        last_reward = [0.0, 0.0]

        def step_sim_once() -> None:
            nonlocal obs, state, sim_tick_count, reset_seed, last_reward, sim_steps_window_count
            should_decide = (sim_tick_count % decision_interval_ticks) == 0
            if should_decide:
                action = {
                    key: _sample_action(obs[key]["action_mask"], action_order, branch_sizes, rng)
                    for key in env.agent_keys
                }
            else:
                action = {
                    key: {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0}
                    for key in env.agent_keys
                }
            obs, reward, done, truncation, _ = env.step(action)
            sim_tick_count += 1
            sim_steps_window_count += 1
            state = env.debug_state()
            last_reward = [float(reward[0]), float(reward[1])]
            if done or truncation:
                reset_seed += 1
                obs, _ = env.reset(seed=reset_seed, options={"ticks_per_step": 1})
                sim_tick_count = 0
                state = env.debug_state()

        running = True
        last_stats = {"fps": 0.0, "arena_w": 1.0, "arena_h": 1.0, "aspect": 1.0}
        while running:
            events = renderer.poll_events()
            if events["quit"]:
                break
            if events["toggle_pause"]:
                paused = not paused
            if events["reset"]:
                reset_seed += 1
                obs, _ = env.reset(seed=reset_seed, options={"ticks_per_step": 1})
                sim_tick_count = 0
                state = env.debug_state()
            if events["step_once"] and paused:
                step_sim_once()

            frame_dt_s = 1.0 / max(1.0, float(fps))
            sim_steps_window_s += frame_dt_s
            if not paused:
                if unthrottled_steps:
                    sim_budget_s = sim_time_budget_fraction * frame_dt_s
                    sim_start_t = time.perf_counter()
                    steps_ran = 0
                    while steps_ran < max_steps_per_frame:
                        elapsed = time.perf_counter() - sim_start_t
                        if steps_ran > 0 and elapsed >= sim_budget_s:
                            break
                        step_sim_once()
                        steps_ran += 1
                else:
                    sim_accum_steps += frame_dt_s * float(sim_limit_steps_per_second)
                    requested_steps = int(sim_accum_steps)
                    steps_to_run = min(requested_steps, max_steps_per_frame)
                    if steps_to_run > 0:
                        frac = sim_accum_steps - float(requested_steps)
                        if best_effort:
                            sim_accum_steps = frac
                        else:
                            sim_accum_steps -= float(steps_to_run)
                        for _ in range(steps_to_run):
                            step_sim_once()
                    if best_effort and requested_steps > steps_to_run:
                        sim_steps_dropped_window_count += requested_steps - steps_to_run

            if sim_steps_window_s >= 0.25:
                sim_steps_actual_sps = float(sim_steps_window_count) / sim_steps_window_s
                sim_steps_dropped_sps = float(sim_steps_dropped_window_count) / sim_steps_window_s
                sim_steps_window_count = 0
                sim_steps_dropped_window_count = 0
                sim_steps_window_s = 0.0

            elixir = state.get("elixir", [0.0, 0.0])
            hud_lines = [
                "[SIM]",
                kv("Time", f"{float(state.get('sim_time_s', 0.0)):.2f}s"),
                kv("Done", str(bool(state.get('done', False)))),
                kv("Truncated", str(bool(state.get('truncation', False)))),
                kv("Decision Hz", f"{float(decision_hz):.2f} (/{decision_interval_ticks} ticks)"),
                kv("Mode", "best-effort" if best_effort else "catch-up"),
                kv("Limit SPS", "unlimited" if unthrottled_steps else f"{float(sim_limit_steps_per_second):.1f}"),
                kv("Actual SPS", f"{sim_steps_actual_sps:.1f} ({sim_steps_actual_sps / float(VIEW_TICK_HZ):.2f}x)"),
                kv("Dropped SPS", f"{sim_steps_dropped_sps:.1f}" if (best_effort and not unthrottled_steps) else "0.0"),
                kv("Target FPS", f"{int(fps)}"),
                kv("Actual FPS", f"{float(last_stats['fps']):.1f}"),
                "",
                "[STATE]",
                kv("Elixir P0", f"{float(elixir[0]):.2f}"),
                kv("Elixir P1", f"{float(elixir[1]):.2f}"),
                kv("Entities", str(len(state.get("entities", [])))),
                kv("Pending", str(len(state.get("pending_spawns", [])))),
                kv("Fireballs", str(len(state.get("fireballs", [])))),
                kv("Logs", str(len(state.get("logs", [])))),
                kv("Reward P0", f"{last_reward[0]:.4f}"),
                kv("Reward P1", f"{last_reward[1]:.4f}"),
                "",
                "[VIEW]",
                kv("Arena Px", f"{int(last_stats['arena_w'])}x{int(last_stats['arena_h'])}"),
                kv("Aspect", f"{float(last_stats['aspect']):.4f}"),
                "",
                "[CONTROLS]",
                "Space: Pause/Resume",
                "N: Step once (paused)",
                "R: Reset",
                "Esc: Quit",
            ]
            last_stats = renderer.draw(state, hud_lines, fps_limit=int(fps))

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        env.close()
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize knockoff_cr_cpp backend state")
    parser.add_argument("--max-sim-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--limit-steps-per-second", type=float, default=None)
    parser.add_argument("--catch-up", action="store_true", default=False)
    parser.add_argument("--decision-hz", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    run_visualizer(
        max_sim_seconds=args.max_sim_seconds,
        seed=args.seed,
        fps=args.fps,
        limit_steps_per_second=args.limit_steps_per_second,
        best_effort=not args.catch_up,
        decision_hz=args.decision_hz,
        max_frames=args.max_frames,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
