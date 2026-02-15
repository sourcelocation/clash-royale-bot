from __future__ import annotations


def test_truncation_happens_with_small_episode(env_factory):
    env = env_factory(tick_hz=10, max_sim_seconds=0.2, seed=1)
    try:
        env.reset(seed=1, options={})
        actions = [
            {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0},
            {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0},
        ]
        t = env.step(actions)
        assert bool(t["truncation"]) is True
        assert bool(t["done"]) is False
    finally:
        del env


def test_done_and_truncation_not_both_true(env):
    transition = env.reset(seed=44, options={})
    for _ in range(250):
        actions = [
            {"wait": 0, "card_selection": 0, "position_region": 0, "position_cell": 0},
            {"wait": 0, "card_selection": 0, "position_region": 0, "position_cell": 0},
        ]
        transition = env.step(actions)
        assert not (bool(transition["done"]) and bool(transition["truncation"]))
        if transition["done"] or transition["truncation"]:
            break
