from __future__ import annotations

import random

import pytest

from .conftest import (
    assert_debug_state_invariants,
    assert_finite_nested,
    branch_sizes,
    sample_action_from_mask,
)


@pytest.mark.slow
def test_long_random_rollout_stability(env):
    spec = env.spec()
    sizes = branch_sizes(spec)
    rng = random.Random(2026)
    transition = env.reset(seed=2026, options={})

    for _ in range(5000):
        actions = [sample_action_from_mask(transition["action_mask"][i], sizes, rng) for i in range(2)]
        transition = env.step(actions)

        assert_finite_nested(transition["reward"])
        for obs in transition["obs"]:
            assert_finite_nested(obs["vector"])

        state = env.debug_state()
        assert_debug_state_invariants(state)

        if transition["done"] or transition["truncation"]:
            transition = env.reset(seed=rng.randint(0, 10_000), options={})

    assert int(spec["n_agents"]) == 2
