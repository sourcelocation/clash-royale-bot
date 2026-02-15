from __future__ import annotations

from training.custom.cpp_env import CppClashEnv

from .conftest import (
    EXPECTED_OBS_VERSION,
    EXPECTED_PROTOCOL_VERSION,
    EXPECTED_SCHEMA_VERSION,
    assert_transition_contract,
    branch_sizes,
)


def test_spec_contract(env):
    spec = env.spec()
    assert spec["protocol_version"] == EXPECTED_PROTOCOL_VERSION
    assert spec["obs_version"] == EXPECTED_OBS_VERSION
    assert spec["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert int(spec["n_agents"]) == 2

    sizes = branch_sizes(spec)
    assert [k for k, _ in sizes] == ["wait", "card_selection", "position_region", "position_cell"]
    assert int(spec["action_mask_size"]) == sum(size for _, size in sizes)


def test_reset_and_step_contract(env):
    spec = env.spec()
    transition = env.reset(seed=123, options={})
    assert_transition_contract(transition, spec)

    actions = [
        {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0},
        {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0},
    ]
    transition = env.step(actions)
    assert_transition_contract(transition, spec)


def test_cpp_env_wrapper_contract():
    env = CppClashEnv(env_id=0, tick_hz=10, max_sim_seconds=15.0, seed=123)
    try:
        obs, info = env.reset(seed=123, options={})
        assert len(obs) == 2
        assert isinstance(info, dict)

        action = {
            k: {
                "wait": 1,
                "card_selection": 0,
                "position_region": 0,
                "position_cell": 0,
            }
            for k in env.agent_keys
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert len(obs) == 2
        assert reward.shape == (2,)
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        assert isinstance(info, dict)

        state = env.debug_state()
        assert state["protocol_version"] == "knockoff_env_debug_v1"
        assert "entities" in state
    finally:
        env.close()
