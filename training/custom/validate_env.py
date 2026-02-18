import argparse
import math

import numpy as np

from .cpp_env import CppClashEnv

EXPECTED_SCHEMA_VERSION = "knockoff_cr_env_v3"

def assert_finite_array(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{name} has non-finite values")


def sample_action(obs: dict, action_space, action_order) -> dict:
    action = {}
    offset = 0
    mask = obs["action_mask"]
    for key in action_order:
        size = int(action_space[key].n)
        branch = mask[offset : offset + size]
        offset += size
        valid = [i for i, v in enumerate(branch) if float(v) > 0.0]
        action[key] = valid[0] if valid else 0
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick-hz", type=int, default=10)
    parser.add_argument("--max-sim-seconds", type=float, default=120.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = CppClashEnv(
        env_id=0,
        tick_hz=int(args.tick_hz),
        max_sim_seconds=float(args.max_sim_seconds),
        seed=int(args.seed),
    )

    try:
        spec = env.spec_data
        action_order = env.action_order
        action_space = env.action_space
        agent_keys = env.agent_keys

        print("[Validate] spec", spec)
        if int(spec.get("n_agents", 0)) != 2:
            raise AssertionError("Expected n_agents=2")
        if str(spec.get("schema_version", "")) != EXPECTED_SCHEMA_VERSION:
            raise AssertionError("Unexpected schema_version")

        reset_options = {"team_controllers": ["external", "human"], "training_mode": True}
        obs_a, _ = env.reset(seed=args.seed, options=reset_options)
        obs_b, _ = env.reset(seed=args.seed, options=reset_options)

        for agent_key in agent_keys:
            vec_a = obs_a[agent_key]["vector"]
            vec_b = obs_b[agent_key]["vector"]
            if vec_a.shape != vec_b.shape:
                raise AssertionError(f"Determinism shape mismatch for {agent_key}")
            if not np.allclose(vec_a, vec_b):
                raise AssertionError(f"Determinism value mismatch for {agent_key}")

        print("[Validate] Determinism check passed.")

        obs, _ = env.reset(seed=args.seed, options=reset_options)
        done_count = 0
        trunc_count = 0
        for i in range(args.steps):
            for agent_key in agent_keys:
                assert_finite_array(f"obs[{agent_key}].vector", obs[agent_key]["vector"])
                assert_finite_array(f"obs[{agent_key}].action_mask", obs[agent_key]["action_mask"])
                assert_finite_array(
                    f"obs[{agent_key}].position_masks_for_all_cards",
                    obs[agent_key]["position_masks_for_all_cards"],
                )
                if obs[agent_key]["action_mask"].shape != env.observation_space[agent_key]["action_mask"].shape:
                    raise AssertionError(f"Mask shape mismatch for {agent_key}")
                if (
                    obs[agent_key]["position_masks_for_all_cards"].shape
                    != env.observation_space[agent_key]["position_masks_for_all_cards"].shape
                ):
                    raise AssertionError(f"Card position mask shape mismatch for {agent_key}")
                if float(np.max(obs[agent_key]["action_mask"])) <= 0.0:
                    raise AssertionError(f"No legal actions available in mask for {agent_key}")

            joint_action = {
                agent_key: sample_action(obs[agent_key], action_space[agent_key], action_order)
                for agent_key in agent_keys
            }
            obs, reward, terminated, truncated, info = env.step(joint_action)

            if reward.shape[0] != len(agent_keys):
                raise AssertionError("Reward vector size mismatch")
            if not np.all(np.isfinite(reward)):
                raise AssertionError("Non-finite reward vector")

            reward_terms = info.get("reward_terms", [])
            if not isinstance(reward_terms, list):
                raise AssertionError("reward_terms must be a list")
            action_debug = info.get("action_debug", [])
            if not isinstance(action_debug, list):
                raise AssertionError("action_debug must be a list")
            pending_total = info.get("pending_spawns_total", None)
            if pending_total is not None and int(pending_total) < 0:
                raise AssertionError("pending_spawns_total must be >= 0")
            pending_by_team = info.get("pending_spawns_team", None)
            if pending_by_team is not None:
                if not isinstance(pending_by_team, list) or len(pending_by_team) != 2:
                    raise AssertionError("pending_spawns_team must be a 2-item list")
            spawns_started = info.get("spawns_started_this_step", None)
            if spawns_started is not None and not isinstance(spawns_started, list):
                raise AssertionError("spawns_started_this_step must be a list")
            spawns_activated = info.get("spawns_activated_this_step", None)
            if spawns_activated is not None and not isinstance(spawns_activated, list):
                raise AssertionError("spawns_activated_this_step must be a list")
            action_applied = info.get("action_applied", [])
            if isinstance(action_applied, list) and len(action_applied) == 2:
                # Team 1 is configured as non-external, so external actions should be ignored.
                if bool(action_applied[1]):
                    raise AssertionError("Team 1 action should not apply when controller is non-external")

            if terminated or truncated:
                done_count += int(terminated)
                trunc_count += int(truncated)
                obs, _ = env.reset(seed=args.seed + i + 1, options=reset_options)

        print(
            f"[Validate] Rollout checks passed. steps={args.steps} done={done_count} trunc={trunc_count}"
        )
        print("[Validate] SUCCESS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
