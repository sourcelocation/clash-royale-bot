from __future__ import annotations

import random

from .conftest import (
    assert_debug_state_invariants,
    assert_finite_nested,
    assert_transition_contract,
    branch_sizes,
    sample_action_from_mask,
)


def test_rollout_invariants(env):
    spec = env.spec()
    sizes = branch_sizes(spec)
    rng = random.Random(12345)

    transition = env.reset(seed=12345, options={})
    assert_transition_contract(transition, spec)

    for _ in range(400):
        actions = [sample_action_from_mask(transition["action_mask"][i], sizes, rng) for i in range(2)]
        transition = env.step(actions)
        assert_transition_contract(transition, spec)

        state = env.debug_state()
        assert_debug_state_invariants(state)

        assert_finite_nested(transition["reward"])
        for obs in transition["obs"]:
            assert_finite_nested(obs["vector"])
            assert_finite_nested(obs["position_masks_for_all_cards"])

        if transition["done"] or transition["truncation"]:
            transition = env.reset(seed=rng.randint(0, 10_000), options={})
