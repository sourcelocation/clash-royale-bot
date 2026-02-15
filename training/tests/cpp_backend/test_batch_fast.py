from __future__ import annotations

import random
from typing import Any

import numpy as np

from .conftest import (
    EXPECTED_OBS_VERSION,
    EXPECTED_PROTOCOL_VERSION,
    EXPECTED_SCHEMA_VERSION,
    assert_debug_state_invariants,
    assert_transition_contract,
    branch_sizes,
    sample_action_from_mask,
)


def _wait_action() -> dict[str, int]:
    return {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0}


def _sample_joint_actions(transition: dict[str, Any], sizes: list[tuple[str, int]], rng: random.Random) -> list[dict[str, int]]:
    return [sample_action_from_mask(transition["action_mask"][team], sizes, rng) for team in range(2)]


def test_batch_contract(batch_env_factory):
    env = batch_env_factory(num_envs=3, seed=123)
    spec = env.spec()
    assert spec["protocol_version"] == EXPECTED_PROTOCOL_VERSION
    assert spec["obs_version"] == EXPECTED_OBS_VERSION
    assert spec["schema_version"] == EXPECTED_SCHEMA_VERSION

    results = env.reset_many(
        seeds=[10, 11, 12],
        options_per_env=[{"ticks_per_step": 1}, {"ticks_per_step": 1}, {"ticks_per_step": 1}],
    )
    assert isinstance(results, list)
    assert len(results) == 3
    for transition in results:
        assert_transition_contract(transition, spec)

    actions_per_env = [[_wait_action(), _wait_action()] for _ in range(3)]
    steps = env.step_many(actions_per_env)
    assert isinstance(steps, list)
    assert len(steps) == 3
    for transition in steps:
        assert_transition_contract(transition, spec)


def test_batch_single_env_equivalence(env_factory, batch_env_factory):
    single = env_factory(seed=777)
    batch = batch_env_factory(num_envs=1, seed=777)
    try:
        seed = 42
        options = {"ticks_per_step": 1}
        t_single = single.reset(seed=seed, options=options)
        t_batch = batch.reset_many(seeds=[seed], options_per_env=[options])[0]
        assert t_single == t_batch

        sizes = branch_sizes(single.spec())
        rng = random.Random(2026)
        for _ in range(120):
            joint = _sample_joint_actions(t_single, sizes, rng)
            t_single = single.step(joint)
            t_batch = batch.step_many([joint])[0]
            assert t_single == t_batch
            if t_single["done"] or t_single["truncation"]:
                break
    finally:
        del single
        del batch


def test_batch_determinism(batch_env_factory):
    env_a = batch_env_factory(num_envs=4, seed=999, num_threads=2)
    env_b = batch_env_factory(num_envs=4, seed=999, num_threads=2)
    try:
        reset_opts = [{"ticks_per_step": 1} for _ in range(4)]
        tr_a = env_a.reset_many(seeds=[100, 101, 102, 103], options_per_env=reset_opts)
        tr_b = env_b.reset_many(seeds=[100, 101, 102, 103], options_per_env=reset_opts)
        assert tr_a == tr_b

        sizes = branch_sizes(env_a.spec())
        rng = random.Random(1234)
        for _ in range(100):
            actions_per_env = [_sample_joint_actions(tr_a[i], sizes, rng) for i in range(4)]
            tr_a = env_a.step_many(actions_per_env)
            tr_b = env_b.step_many(actions_per_env)
            assert tr_a == tr_b
    finally:
        del env_a
        del env_b


def test_batch_parallel_stability(batch_env_factory):
    num_envs = 6
    env = batch_env_factory(num_envs=num_envs, seed=123, num_threads=3)
    reset_opts = [{"ticks_per_step": 1} for _ in range(num_envs)]
    transitions = env.reset_many(seeds=[123 + i for i in range(num_envs)], options_per_env=reset_opts)
    sizes = branch_sizes(env.spec())
    rng = random.Random(7)

    for _ in range(300):
        actions_per_env = [_sample_joint_actions(transitions[i], sizes, rng) for i in range(num_envs)]
        transitions = env.step_many(actions_per_env)
        for transition in transitions:
            reward = np.asarray(transition["reward"], dtype=np.float64)
            assert np.isfinite(reward).all()
        states = env.debug_state_many()
        assert len(states) == num_envs
        for state in states:
            assert_debug_state_invariants(state)


def test_batch_step_many_discrete_equivalence(batch_env_factory):
    num_envs = 4
    env_dict = batch_env_factory(num_envs=num_envs, seed=123, num_threads=2)
    env_discrete = batch_env_factory(num_envs=num_envs, seed=123, num_threads=2)
    try:
        reset_opts = [{"ticks_per_step": 1} for _ in range(num_envs)]
        tr_dict = env_dict.reset_many(seeds=[123 + i for i in range(num_envs)], options_per_env=reset_opts)
        tr_discrete = env_discrete.reset_many(seeds=[123 + i for i in range(num_envs)], options_per_env=reset_opts)
        assert tr_dict == tr_discrete

        sizes = branch_sizes(env_dict.spec())
        action_order = [key for key, _ in sizes]
        rng = random.Random(9)
        for _ in range(100):
            actions_per_env = [_sample_joint_actions(tr_dict[i], sizes, rng) for i in range(num_envs)]
            packed = np.zeros((num_envs, 2, len(action_order)), dtype=np.int32)
            for env_i in range(num_envs):
                for team in range(2):
                    for b_idx, key in enumerate(action_order):
                        packed[env_i, team, b_idx] = int(actions_per_env[env_i][team][key])

            tr_dict = env_dict.step_many(actions_per_env)
            tr_discrete = env_discrete.step_many_discrete(packed)
            assert tr_dict == tr_discrete
    finally:
        del env_dict
        del env_discrete


def test_batch_step_many_packed_equivalence(batch_env_factory):
    num_envs = 3
    env_dict = batch_env_factory(num_envs=num_envs, seed=321, num_threads=2)
    env_packed = batch_env_factory(num_envs=num_envs, seed=321, num_threads=2)
    try:
        reset_opts = [{"ticks_per_step": 1} for _ in range(num_envs)]
        tr_dict = env_dict.reset_many(seeds=[100 + i for i in range(num_envs)], options_per_env=reset_opts)
        tr_packed = env_packed.reset_many(seeds=[100 + i for i in range(num_envs)], options_per_env=reset_opts)
        assert tr_dict == tr_packed

        sizes = branch_sizes(env_dict.spec())
        action_order = [key for key, _ in sizes]
        rng = random.Random(11)
        for _ in range(60):
            actions_per_env = [_sample_joint_actions(tr_dict[i], sizes, rng) for i in range(num_envs)]
            packed_actions = np.zeros((num_envs, 2, len(action_order)), dtype=np.int32)
            for env_i in range(num_envs):
                for team in range(2):
                    for b_idx, key in enumerate(action_order):
                        packed_actions[env_i, team, b_idx] = int(actions_per_env[env_i][team][key])

            tr_dict = env_dict.step_many(actions_per_env)
            payload = env_packed.step_many_packed(packed_actions)
            obs = np.asarray(payload["obs"], dtype=np.float32)
            action_mask = np.asarray(payload["action_mask"], dtype=np.float32)
            card_masks = np.asarray(payload["card_position_masks"], dtype=np.float32)
            reward = np.asarray(payload["reward"], dtype=np.float32)
            done = np.asarray(payload["done"], dtype=np.uint8).astype(np.bool_, copy=False)
            trunc = np.asarray(payload["truncation"], dtype=np.uint8).astype(np.bool_, copy=False)
            winner = np.asarray(payload["winner"], dtype=np.int8)
            tr_packed = []
            for env_i in range(num_envs):
                tr_packed.append(
                    {
                        "obs": [
                            {
                                "vector": obs[env_i, 0].tolist(),
                                "position_masks_for_all_cards": card_masks[env_i, 0].tolist(),
                            },
                            {
                                "vector": obs[env_i, 1].tolist(),
                                "position_masks_for_all_cards": card_masks[env_i, 1].tolist(),
                            },
                        ],
                        "action_mask": [action_mask[env_i, 0].tolist(), action_mask[env_i, 1].tolist()],
                        "reward": reward[env_i].tolist(),
                        "done": bool(done[env_i]),
                        "truncation": bool(trunc[env_i]),
                    }
                )
            for env_i in range(num_envs):
                assert tr_dict[env_i]["done"] == tr_packed[env_i]["done"]
                assert tr_dict[env_i]["truncation"] == tr_packed[env_i]["truncation"]
                np.testing.assert_allclose(np.asarray(tr_dict[env_i]["reward"]), np.asarray(tr_packed[env_i]["reward"]))
                np.testing.assert_allclose(
                    np.asarray(tr_dict[env_i]["obs"][0]["vector"]), np.asarray(tr_packed[env_i]["obs"][0]["vector"])
                )
                np.testing.assert_allclose(
                    np.asarray(tr_dict[env_i]["obs"][1]["vector"]), np.asarray(tr_packed[env_i]["obs"][1]["vector"])
                )
                np.testing.assert_allclose(
                    np.asarray(tr_dict[env_i]["action_mask"][0]), np.asarray(tr_packed[env_i]["action_mask"][0])
                )
                np.testing.assert_allclose(
                    np.asarray(tr_dict[env_i]["action_mask"][1]), np.asarray(tr_packed[env_i]["action_mask"][1])
                )
                if tr_packed[env_i]["done"] and (not tr_packed[env_i]["truncation"]):
                    assert int(winner[env_i]) in (-1, 0, 1)
    finally:
        del env_dict
        del env_packed
